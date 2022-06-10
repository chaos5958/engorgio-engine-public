#include "neural_enhancer_server.h"
#include <algorithm>
#include <chrono>
#include <cassert>
#include <thread>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>


using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

const unsigned NUM_GRPC_HANDLER_THREADS = 4;

NeuralEnhancerServer::NeuralEnhancerServer(const std::string bind_address) : bind_address_(bind_address) {
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
}

void NeuralEnhancerServer::LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer)
{
    neural_enhancer_ = neural_enhancer;
}


void NeuralEnhancerServer::HandleRpcs(int thread_idx, grpc::ServerCompletionQueue *cq) 
{
    std::cerr << "Starting handling RPC at " << thread_idx << std::endl;
    new CallDataProcess(this, cq);
    new CallDataGetGPUs(this, cq);
    new CallDataLoadEngine(this, cq);
    new CallDataFinished(this, cq);
    new CallDataTerminate(this, cq);
    void* tag;
    bool ok;
    while (true) {
        bool ret = cq->Next(&tag, &ok);
        if (ok == false || ret == false)
        {
            return;
        }
        static_cast<CallDataBase*>(tag)->Proceed();
    }
}

void NeuralEnhancerServer::run() {
    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(1024*1024*1024); // 1 GB
    builder.SetMaxSendMessageSize(1024*1024*1024); // 1 GB
    builder.AddListeningPort(bind_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    
    for (unsigned i = 0; i < NUM_GRPC_HANDLER_THREADS; i++)
        cqs_.push_back(builder.AddCompletionQueue());
    server_ = builder.BuildAndStart();
    std::cout << "Server listening on " << bind_address_ << std::endl;

    std::vector<std::thread> handler_threads;
    for (unsigned i = 0; i < NUM_GRPC_HANDLER_THREADS; i++)
        handler_threads.push_back(std::thread(&NeuralEnhancerServer::HandleRpcs, this, i, cqs_[i].get()));
        
    server_->Wait();

    for (unsigned i = 0; i < NUM_GRPC_HANDLER_THREADS; i++)
        handler_threads[i].join();
}

static EngorgioModel *deserializeEngorgioModel(const engorgio::grpc::EngorgioModel &grpc_model) {
    uint8_t *model_buffer = new uint8_t[grpc_model.size()];
    memcpy(model_buffer, grpc_model.buf().c_str(), grpc_model.size());
    return new EngorgioModel(model_buffer, grpc_model.size(), grpc_model.name(), grpc_model.scale());
}

static EngorgioFrame *deserializeEngorgioFrame(const engorgio::grpc::EngorgioFrame &grpc_frame) {
    auto frame = new EngorgioFrame(grpc_frame.width(), grpc_frame.height());
    memcpy(frame->rgb_buffer, grpc_frame.rgb_buffer().c_str(), frame->width * frame->height * 3 * sizeof(uint8_t));
    frame->width_alloc = grpc_frame.width_alloc();
    frame->height_alloc = grpc_frame.height_alloc();
    
    frame->frame_type = grpc_frame.frame_type();
    frame->residual_size = grpc_frame.residual_size();
    frame->current_video_frame = grpc_frame.current_video_frame();
    frame->current_super_frame = grpc_frame.current_super_frame();


    frame->stream_id = grpc_frame.stream_id();
    frame->is_sorted = grpc_frame.is_sorted();
    frame->diff_residual = grpc_frame.diff_residual();
    frame->prev_total_residual = grpc_frame.prev_total_residual();
    frame->offset = grpc_frame.offset();
    
    return frame;
}

grpc::Status 
NeuralEnhancerServer::Process(grpc::ServerContext *context, const engorgio::grpc::ProcessRequest *request, engorgio::grpc::ProcessReply *reply) {
    // std::cerr << "NeuralEnhancerServer::Process" << std::endl;
    std::vector<EngorgioFrame *> frame_vec;
    std::transform(request->frames().begin(), request->frames().end(), std::back_inserter(frame_vec), 
                    [](const engorgio::grpc::EngorgioFrame &grpc_frame) -> EngorgioFrame * { return deserializeEngorgioFrame(grpc_frame); });

    std::chrono::microseconds dur(request->time_point());
    std::chrono::time_point<std::chrono::system_clock> deadline(dur);

    EngorgioModel *model = deserializeEngorgioModel(request->model());

    neural_enhancer_->Process(
        request->stream_id(),
        model,
        nullptr,
        frame_vec,
        deadline,
        request->free_model());
    // std::cerr << "NeuralEnhancerServer::Process OK" << std::endl;
    return Status::OK;
}

grpc::Status
NeuralEnhancerServer::GetGPUs(grpc::ServerContext *context, const engorgio::grpc::GetGPUsRequest *request, engorgio::grpc::GetGPUsReply *reply) {
    // std::cerr << "NeuralEnhancerServer::GetGPUs" << std::endl;
    reply->set_length(neural_enhancer_->GetGPUs());
    // std::cerr << "NeuralEnhancerServer::GetGPUs OK" << std::endl;
    return Status::OK;
}

grpc::Status
NeuralEnhancerServer::LoadEngine(grpc::ServerContext *context, const engorgio::grpc::LoadEngineRequest *request, engorgio::grpc::LoadEngineReply *reply) {
    // std::cerr << "NeuralEnhancerServer::LoadEngine" << std::endl;
    bool result = neural_enhancer_->LoadEngine(deserializeEngorgioModel(request->model()));
    reply->set_result(result);
    // std::cerr << "NeuralEnhancerServer::LoadEngine OK" << std::endl;
    return Status::OK;
}
    

grpc::Status
NeuralEnhancerServer::Finished(grpc::ServerContext *context, const engorgio::grpc::FinishedRequest *request, engorgio::grpc::FinishedReply *reply) {
    // std::cerr << "NeuralEnhancerServer::Finished" << std::endl;
    bool result = false;
    int num_frames = request->num_frames();
    
    switch(neural_enhancer_->etype_)
    {
        case EnhancerType::kEngorgio:
            if (!neural_enhancer_->jpeg_engine_)
            {
                result = neural_enhancer_->infer_engine_->Finished();
                break;
            }
            else
            {
                assert(num_frames != 0);
                if (neural_enhancer_->jpeg_engine_->save_image_) 
                    result = neural_enhancer_->jpeg_engine_->SaveFinished(num_frames);
                else   
                    result = neural_enhancer_->jpeg_engine_->EncodeFinished(num_frames);            
                break;
            }
    
        case EnhancerType::kPerFrame:
        case EnhancerType::kSelective:
        default:
            return Status::CANCELLED;
    }
    
    reply->set_result(result);
    // std::cerr << "NeuralEnhancerServer::Finished OK - " << result << std::endl;
    return Status::OK;
}

grpc::Status
NeuralEnhancerServer::Terminate(grpc::ServerContext *context, const engorgio::grpc::TerminateRequest *request, engorgio::grpc::TerminateReply *reply) {
    std::cerr << "Shutting down server..." << std::endl;
    
    const std::chrono::milliseconds waitDuration = std::chrono::milliseconds(50);
    const std::chrono::time_point<std::chrono::system_clock> deadline = std::chrono::system_clock::now() + waitDuration;
    server_->Shutdown(deadline);
        
    delete neural_enhancer_;

    return Status::OK;
}
