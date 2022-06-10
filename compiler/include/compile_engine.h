#pragma once

#include <vector>
#include <string>
#include <functional>
#include <condition_variable>
#include <queue>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <chrono>
#include <filesystem>

#include "compile_common.h"
#include "NvInfer.h"
#include "argsParser.h"
#include "common.h"


struct CompileEngineProfile {
    int num_threads;
    std::string log_dir; 
    bool save_log = false;
};

enum class OptType : int
{
	kOpt,
    kJoin
};

struct OptEvent{
    OptType type;
    int stream_id;
    OnnxModel *onnx_model;
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

struct OptDNN
{
    int stream_id;
    nvinfer1::IHostMemory* opt_dnn;
};

struct CompileLog
{
    std::string task;
    int thread_id;
    std::string model_name;
    double latency;
};

// TODO: multiple GPU support (use cudaSetDevice())
// TODO: logging (work type (preopt, opt), model name, latency)
class CompileEngine
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    
private:    
    std::thread* preopt_worker_;
    // std::map<std::string, nvinfer1::ICudaEngine*> preopt_dnns_;
    std::map<std::string, nvinfer1::IHostMemory*> preopt_dnns_;
    std::deque<OptEvent> preopt_events_;
    std::mutex preopt_emutex_;
    std::shared_mutex preopt_dmutex;

    int num_threads_;
    // std::map<int, nvinfer1::IHostMemory*> opt_dnns_;
    std::deque<OptDNN *> opt_dnns_;
    std::vector<std::thread*> opt_workers_;
    std::deque<OptEvent> opt_events_;
    std::mutex opt_emutex_;
    std::shared_mutex opt_dmutex_;

    // log
    bool save_log_;
    std::filesystem::path log_dir_;
    std::deque<CompileLog*> logs_;
    std::mutex log_mutex_;

    void PreOptHandler();
    bool PreOptInternal(OptEvent &event);
    void OptHandler(int thread_id);
    bool OptInternal(OptEvent &event, int thread_id);
    static nvinfer1::IHostMemory *Refit(nvinfer1::INetworkDefinition* network, nvinfer1::IHostMemory* trt_model);
public:
    CompileEngine(CompileEngineProfile &cprofile);
    ~CompileEngine();
    void PreOptimize(OnnxModel *onnx_model);
    void Optimize(int stream_id, OnnxModel *onnx_model);
    void GetOptDNNs(std::deque<OptDNN*> &results);
    // nvinfer1::IHostMemory* Optimize(int stream_id, OnnxModel *onnx_model);
    void Register(std::string &name, void *buf, size_t size);
    void Save();

    bool PreOptExists(std::string dnn_name); // for debugging
    int GetOptSize(); // for debugging
};

