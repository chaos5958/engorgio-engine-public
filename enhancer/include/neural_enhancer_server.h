#pragma once
#include <string>

#include <grpcpp/grpcpp.h>
#include "enhancer.grpc.pb.h"
#include "neural_enhancer.h"

class NeuralEnhancerServer final : public engorgio::grpc::GrpcNeuralEnhancer::AsyncService {
private:
    std::string bind_address_;
    std::unique_ptr<grpc::Server> server_ {nullptr};
    std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> cqs_;
    NeuralEnhancer *neural_enhancer_{nullptr};

    void HandleRpcs(int thread_idx, grpc::ServerCompletionQueue *cq);

public:
    NeuralEnhancerServer(const std::string bind_address);
    void run();
    void LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer);

    grpc::Status Process(grpc::ServerContext *context, const engorgio::grpc::ProcessRequest *request, engorgio::grpc::ProcessReply *reply) override;
    grpc::Status GetGPUs(grpc::ServerContext *context, const engorgio::grpc::GetGPUsRequest *request, engorgio::grpc::GetGPUsReply *reply) override;
    grpc::Status LoadEngine(grpc::ServerContext *context, const engorgio::grpc::LoadEngineRequest *request, engorgio::grpc::LoadEngineReply *reply) override;
    grpc::Status Finished(grpc::ServerContext *context, const engorgio::grpc::FinishedRequest *request, engorgio::grpc::FinishedReply *reply) override;
    grpc::Status Terminate(grpc::ServerContext *context, const engorgio::grpc::TerminateRequest *request, engorgio::grpc::TerminateReply *reply) override;
};

class CallDataBase
{
protected:
    virtual void WaitForRequest() = 0;
    virtual void HandleRequest() = 0;
public:
    virtual void Proceed() = 0;
    CallDataBase() {}
    virtual ~CallDataBase() {}
};


template < class RequestType, class ReplyType>
class CallDataT : CallDataBase
{
protected:
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;

    NeuralEnhancerServer* service_;
    grpc::ServerCompletionQueue* completionQueue_;
    RequestType request_;
    ReplyType reply_;
    grpc::ServerAsyncResponseWriter<ReplyType> responder_;
    grpc::ServerContext serverContext_;

    // When we handle a request of this type, we need to tell
    // the completion queue to wait for new requests of the same type.
    virtual void AddNextToCompletionQueue() = 0;
public:
    CallDataT(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) :
        status_(CREATE),
        service_(service),
        completionQueue_(completionQueue),
        responder_(&serverContext_)
    {
    }
    virtual void Proceed() override
    {
        if (status_ == CREATE)
        {
            status_ = PROCESS;
            WaitForRequest();
        }
        else if (status_ == PROCESS)
        {
            AddNextToCompletionQueue();
            HandleRequest();
            status_ = FINISH;
            responder_.Finish(reply_, grpc::Status::OK, this);
        }
        else
        {
            // We're done! Self-destruct!
            if (status_ != FINISH)
            {
                // Log some error message
            }
            delete this;
        }
    }
};

class CallDataProcess : CallDataT<engorgio::grpc::ProcessRequest, engorgio::grpc::ProcessReply>
{
public:
    CallDataProcess(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) : CallDataT(service, completionQueue) {
        Proceed();
    }
protected:
    virtual void AddNextToCompletionQueue() override {
        new CallDataProcess(service_, completionQueue_);
    }
    virtual void WaitForRequest() override {
        service_->RequestProcess(&serverContext_, &request_, &responder_, completionQueue_, completionQueue_, this);
    }
    virtual void HandleRequest() override {
        service_->Process(&serverContext_, &request_, &reply_);
    }
};

class CallDataGetGPUs : CallDataT<engorgio::grpc::GetGPUsRequest, engorgio::grpc::GetGPUsReply>
{
public:
    CallDataGetGPUs(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) : CallDataT(service, completionQueue) {
        Proceed();
    }
protected:
    virtual void AddNextToCompletionQueue() override {
        new CallDataGetGPUs(service_, completionQueue_);
    }
    virtual void WaitForRequest() override {
        service_->RequestGetGPUs(&serverContext_, &request_, &responder_, completionQueue_, completionQueue_, this);
    }
    virtual void HandleRequest() override {
        service_->GetGPUs(&serverContext_, &request_, &reply_);
    }
};

class CallDataLoadEngine : CallDataT<engorgio::grpc::LoadEngineRequest, engorgio::grpc::LoadEngineReply>
{
public:
    CallDataLoadEngine(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) : CallDataT(service, completionQueue) {
        Proceed();
    }
protected:
    virtual void AddNextToCompletionQueue() override {
        new CallDataLoadEngine(service_, completionQueue_);
    }
    virtual void WaitForRequest() override {
        service_->RequestLoadEngine(&serverContext_, &request_, &responder_, completionQueue_, completionQueue_, this);
    }
    virtual void HandleRequest() override {
        service_->LoadEngine(&serverContext_, &request_, &reply_);
    }
};

class CallDataFinished : CallDataT<engorgio::grpc::FinishedRequest, engorgio::grpc::FinishedReply>
{
public:
    CallDataFinished(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) : CallDataT(service, completionQueue) {
        Proceed();
    }
protected:
    virtual void AddNextToCompletionQueue() override {
        new CallDataFinished(service_, completionQueue_);
    }
    virtual void WaitForRequest() override {
        service_->RequestFinished(&serverContext_, &request_, &responder_, completionQueue_, completionQueue_, this);
    }
    virtual void HandleRequest() override {
        service_->Finished(&serverContext_, &request_, &reply_);
    }
};

class CallDataTerminate : CallDataT<engorgio::grpc::TerminateRequest, engorgio::grpc::TerminateReply>
{
public:
    CallDataTerminate(NeuralEnhancerServer* service, grpc::ServerCompletionQueue* completionQueue) : CallDataT(service, completionQueue) {
        Proceed();
    }
protected:
    virtual void AddNextToCompletionQueue() override {
        new CallDataTerminate(service_, completionQueue_);
    }
    virtual void WaitForRequest() override {
        service_->RequestTerminate(&serverContext_, &request_, &responder_, completionQueue_, completionQueue_, this);
    }
    virtual void HandleRequest() override {
        service_->Terminate(&serverContext_, &request_, &reply_);
    }
};