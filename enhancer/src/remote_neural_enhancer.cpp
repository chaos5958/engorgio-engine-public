
#include <iostream>
#include <sstream>
#include <algorithm>
#include "remote_neural_enhancer.h"
#include "enhancer.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

static void throwGrpcException(grpc::Status &status) {
    std::stringstream ss;
    ss << "RPC failed: " << status.error_code() << ": " << status.error_message();
    throw std::runtime_error(ss.str());
}

RemoteNeuralEnhancer::RemoteNeuralEnhancer(InferEngineProfile &iprofile, const std::vector<std::string> &targets) : NeuralEnhancer(iprofile, false)
{
    jpeg_engine_ = nullptr;
    infer_engine_ = nullptr;
    etype_ = EnhancerType::kEngorgio;

    auto cargs = grpc::ChannelArguments();
    cargs.SetMaxReceiveMessageSize(1024*1024*1024); // 1 GB
    cargs.SetMaxSendMessageSize(1024*1024*1024); // 1 GB

    // Connect to given hosts
    for (auto it = targets.begin(); it != targets.end(); ++it)
    {
        std::cerr << "Creating channel with " << *it << std::endl;
        auto channel = grpc::CreateCustomChannel(*it, grpc::InsecureChannelCredentials(), cargs);
        stub_list_.push_back(engorgio::grpc::GrpcNeuralEnhancer::NewStub(channel));
    }
    std::cerr << "Channel OK" << std::endl;
    async_rpc_handler_ = std::thread(&RemoteNeuralEnhancer::AsyncCompleteRpc, this);
}

RemoteNeuralEnhancer::~RemoteNeuralEnhancer()
{
    // std::cerr << "Terminating remote neural enhancers..." << std::endl;
    for (auto it = stub_list_.begin(); it != stub_list_.end(); ++it)
    {
        engorgio::grpc::TerminateRequest request;
        engorgio::grpc::TerminateReply reply;
        ClientContext context;

        (*it)->Terminate(&context, request, &reply);
    }

    if (infer_engine_)
        delete infer_engine_;
    if (jpeg_engine_)
        delete jpeg_engine_;
    stub_list_.clear();

    cq_.Shutdown();
    async_rpc_handler_.join();
}

static engorgio::grpc::EngorgioModel *serializeEngorgioModel(const EngorgioModel *model) {
    engorgio::grpc::EngorgioModel *grpc_model = new engorgio::grpc::EngorgioModel();
    grpc_model->set_buf(model->buf_, model->size_);
    grpc_model->set_name(model->name_);
    grpc_model->set_size(model->size_);
    grpc_model->set_scale(model->scale_);
    return grpc_model;
}

static void serializeEngorgioFrame(engorgio::grpc::EngorgioFrame *grpc_frame, const EngorgioFrame *frame) {
    grpc_frame->set_rgb_buffer(frame->rgb_buffer, sizeof(uint8_t) * frame->width * frame->height * 3);
    grpc_frame->set_width(frame->width);
    grpc_frame->set_height(frame->height);
    grpc_frame->set_width_alloc(frame->width_alloc);
    grpc_frame->set_height_alloc(frame->height_alloc);

    grpc_frame->set_frame_type(frame->frame_type);
    grpc_frame->set_residual_size(frame->residual_size);
    grpc_frame->set_current_video_frame(frame->current_video_frame);
    grpc_frame->set_current_super_frame(frame->current_super_frame);

    grpc_frame->set_stream_id(frame->stream_id);
    grpc_frame->set_is_sorted(frame->is_sorted);
    grpc_frame->set_diff_residual(frame->diff_residual);
    grpc_frame->set_prev_total_residual(frame->prev_total_residual);
    grpc_frame->set_offset(frame->offset);
}


void RemoteNeuralEnhancer::Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &deadline, bool free_model)
{
    // std::cerr << "RemoteNeuralEnhancer::Process" << std::endl;
    engorgio::grpc::ProcessRequest request;
    engorgio::grpc::ProcessReply reply;
    request.set_stream_id(stream_id);
    request.set_allocated_model(serializeEngorgioModel(model));

    size_t frames_len = frames.size();
    for (size_t i = 0; i < frames_len; i++)
    {
        auto grpc_frame = request.add_frames();
        serializeEngorgioFrame(grpc_frame, frames[i]);
    }

    auto deadline_us = std::chrono::time_point_cast<std::chrono::microseconds>(deadline).time_since_epoch().count();
    request.set_time_point(deadline_us);
    request.set_free_model(free_model);

    // do round-robin scheduling to each instances
    AsyncProcessCall *call = new AsyncProcessCall;
    std::copy(frames.begin(), frames.end(), std::back_inserter(call->frames_to_free));
    call->frame_pool = framepool;
    call->response_reader = stub_list_[next_rr_stub_id_]->PrepareAsyncProcess(&call->context, request, &cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, (void*)call);

    next_rr_stub_id_ = (next_rr_stub_id_ + 1) % stub_list_.size();
    // std::cerr << "RemoteNeuralEnhancer::Process OK" << std::endl;
}

bool RemoteNeuralEnhancer::Finished(int num_frames)
{
    // std::cerr << "RemoteNeuralEnhancer::Finished" << std::endl;
    for (auto it = stub_list_.begin(); it != stub_list_.end(); ++it)
    {
        engorgio::grpc::FinishedRequest request;
        engorgio::grpc::FinishedReply reply;
        ClientContext context;

        request.set_num_frames(num_frames);
        Status status = (*it)->Finished(&context, request, &reply);
        if (status.ok()) 
        {
            if (!reply.result())
                return false;
        }
        else
        {
            throwGrpcException(status);
        }
    }
    return true;
}

bool RemoteNeuralEnhancer::LoadEngine(EngorgioModel *model)
{
    // std::cerr << "RemoteNeuralEnhancer::LoadEngine" << std::endl;
    for (auto it = stub_list_.begin(); it != stub_list_.end(); ++it)
    {
        engorgio::grpc::LoadEngineRequest request;
        engorgio::grpc::LoadEngineReply reply;
        ClientContext context;

        request.set_allocated_model(serializeEngorgioModel(model));
        Status status = (*it)->LoadEngine(&context, request, &reply);
        if (status.ok()) 
        {
            if (!reply.result())
                return false;
        }
        else
        {
            throwGrpcException(status);
        }
    }
    // std::cerr << "RemoteNeuralEnhancer::LoadEngine OK" << std::endl;
    return true;
}

unsigned int RemoteNeuralEnhancer::GetGPUs()
{
    unsigned int num_gpus = 0;
    for (auto it = stub_list_.begin(); it != stub_list_.end(); ++it)
    {
        engorgio::grpc::GetGPUsRequest request;
        engorgio::grpc::GetGPUsReply reply;
        ClientContext context;

        Status status = (*it)->GetGPUs(&context, request, &reply);
        if (status.ok()) {
            num_gpus += reply.length();
        } else {
            throwGrpcException(status);
        }
    }

    return num_gpus;
}

void RemoteNeuralEnhancer::AsyncCompleteRpc() {
    void* got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue "cq".
    while (cq_.Next(&got_tag, &ok)) {
        // The tag in this example is the memory location of the call object
        AsyncProcessCall* call = static_cast<AsyncProcessCall*>(got_tag);

        GPR_ASSERT(ok);

        // free frames
        call->frame_pool->FreeFrames(call->frames_to_free);

        // Once we're complete, deallocate the call object.
        delete call;
    }
}