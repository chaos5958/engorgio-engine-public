#pragma once 
#include <string>
#include <vector>
#include <thread>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#include "neural_enhancer.h"
#include "enhancer.grpc.pb.h"

struct AsyncProcessCall {
    engorgio::grpc::ProcessReply reply;
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<engorgio::grpc::ProcessReply>> response_reader;
    std::vector<EngorgioFrame *> frames_to_free;
    EngorgioFramePool *frame_pool;
};

// RemoteNeuralEnhancer is a client-side implementation to InferEngine
class RemoteNeuralEnhancer : public NeuralEnhancer
{
private:
    std::vector<std::unique_ptr<engorgio::grpc::GrpcNeuralEnhancer::Stub>> stub_list_;
    size_t next_rr_stub_id_ {0};
    grpc::CompletionQueue cq_;
    std::thread async_rpc_handler_;

public:
    RemoteNeuralEnhancer(InferEngineProfile &iprofile, const std::vector<std::string> &targets);

    virtual ~RemoteNeuralEnhancer();

    void Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &dealine, bool free_model = true);

    bool LoadEngine(EngorgioModel *model);
    bool Finished(int num_frames = 0);
    unsigned int GetGPUs();
    virtual bool isRemote() const { return true; }
    void AsyncCompleteRpc();
};