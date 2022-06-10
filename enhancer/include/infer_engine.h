#pragma once 

#include <vector>
#include <filesystem>
#include <thread>
#include <mutex>
#include <deque>
#include <nvml.h>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuDeviceMemory.h"
#include "cuHostMemoryV2.h"
#include "common.h"
#include "enhancer_common.h"
#include "control_common.h"
#include "ipp.h"
#include "ippcc.h"
#include "ippi.h"

const int MAX_CUDA_STREAMS = 3; 
const int MAX_HLOAD_STREAMS = 1;
const int NUM_DEPULICATES = 2;

class JPEGEncodeEngine;
class libvpxEngine;

enum class InferEngineType : int
{
    kEngorgio = 0,
    kBaseline1 = 1,
    kBaseline2 = 2, // deprecated
    kPerFrame = 3,
    kSelective = 4,
};

enum class InferEventType : int
{
    kQuery = 0,
    kJoin = 1,
};

struct InferEvent
{
    InferEventType type;
    int query_id;
};

struct InferEngineProfile
{
    InferEngineType type;
    bool save_log;
    unsigned int num_gpus;
    std::filesystem::path log_dir;
    std::chrono::system_clock::time_point start;
    std::vector<std::vector<int>> thread_indexes;

    InferEngineProfile(InferEngineType type_, bool save_log_, std::chrono::system_clock::time_point start_ = std::chrono::system_clock::now(), 
                        std::vector<std::vector<int>> thread_indexes_ = {}, std::filesystem::path save_dir_=std::filesystem::path(""))
    {
        type = type_;
        save_log = save_log_;
        log_dir = save_dir_;
        start = start_;

        nvmlReturn_t ret;
        ret = nvmlInit();
        if (ret !=  NVML_SUCCESS)
            throw std::runtime_error("nvmlInit failed");
        ret = nvmlDeviceGetCount(&num_gpus);
        if (ret !=  NVML_SUCCESS)
            throw std::runtime_error("nvmlInit failed");    

        if (thread_indexes_.size() == 0)
        {
            std::vector<int> gpu_thread_indexes;
            for (unsigned int i = 0; i < num_gpus; i++)
            {
                gpu_thread_indexes.push_back(i * num_gpus);
                gpu_thread_indexes.push_back(i * num_gpus + 1);
                gpu_thread_indexes.push_back(i * num_gpus + 2);
                thread_indexes.push_back(gpu_thread_indexes);
                gpu_thread_indexes.clear();
            }
        }
        else
        {
            thread_indexes = thread_indexes_;
        }
    }

    InferEngineProfile(InferEngineType type_, bool save_log_, unsigned int num_gpus_, std::chrono::system_clock::time_point start_ = std::chrono::system_clock::now(),
                    std::vector<std::vector<int>> thread_indexes_ = {}, std::filesystem::path save_dir_=std::filesystem::path(""))
    {
        type = type_;
        save_log = save_log_;
        log_dir = save_dir_;
        num_gpus = num_gpus_;
        start = start_;

        if (thread_indexes_.size() == 0)
        {
            std::vector<int> gpu_thread_indexes;
            for (unsigned int i = 0; i < num_gpus; i++)
            {
                gpu_thread_indexes.push_back(i * num_gpus);
                gpu_thread_indexes.push_back(i * num_gpus + 1);
                gpu_thread_indexes.push_back(i * num_gpus + 2);
                thread_indexes.push_back(gpu_thread_indexes);
                gpu_thread_indexes.clear();
            }
        }
        else
        {
            thread_indexes = thread_indexes_;
        }
    }
};



struct RefitEngine
{
    // std::map<int, std::deque<ICudaEngine*>> cuda_engines_per_gpu;
    // std::map<int, std::unique_ptr<std::mutex>> mutexes_per_gpu;
    std::vector<std::deque<ICudaEngine*>> cuda_engines_per_gpu;
    std::vector<std::mutex*> mutexes_per_gpu;

    RefitEngine(int num_gpus) 
    {
        for (int i = 0; i < num_gpus; i++)
        {
            cuda_engines_per_gpu.push_back(std::deque<ICudaEngine*>());
            mutexes_per_gpu.push_back(new std::mutex());
        }
    }

    ~RefitEngine()
    {
        ICudaEngine *engine;
        for (auto & cuda_engines : cuda_engines_per_gpu)
        {
            while(cuda_engines.size() > 0)
            {
                engine = cuda_engines.front();
                cuda_engines.pop_front();
                engine->destroy();
            }
        }
        for (auto & mutex: mutexes_per_gpu)
            delete mutex;
    }
};


// TODO: add constructure
struct InferQueryLog{
    int stream_id;
    std::chrono::system_clock::time_point refit_start, refit_end;
    std::chrono::system_clock::time_point host_start, host_end;
    std::chrono::system_clock::time_point device_start, device_end;
    std::chrono::system_clock::time_point infer_start, infer_end;
    std::chrono::system_clock::time_point unload_start, unload_end;
    std::vector<int> video_indexes;
    std::vector<int> super_indexes;
    std::chrono::system_clock::time_point deadline;
    bool has_deadline;
};

struct InferFrameLog{
    int video_index, super_index;
    std::chrono::system_clock::time_point refit_start, refit_end;
    std::chrono::system_clock::time_point host_start, host_end;
    std::chrono::system_clock::time_point device_start, device_end;
    std::chrono::system_clock::time_point infer_start, infer_end;
    std::chrono::system_clock::time_point unload_start, unload_end;
    std::chrono::system_clock::time_point deadline;
    bool has_deadline;

    InferFrameLog(InferQueryLog &query_log, int i)
    {
        video_index = query_log.video_indexes[i];
        super_index = query_log.super_indexes[i];
        refit_start = query_log.refit_start;
        refit_end = query_log.refit_end;
        host_start = query_log.host_start;
        host_end = query_log.host_end;
        device_start = query_log.device_start;
        device_end = query_log.device_end;
        infer_start = query_log.infer_start;
        infer_end = query_log.infer_end;
        unload_start = query_log.unload_start;
        unload_end = query_log.unload_end;

        if (query_log.has_deadline)
        {
            has_deadline = true;
            deadline = query_log.deadline;
        }
    }
};

struct QueryContext
{
    // qeury
    int stream_id;
    EngorgioModel *model;
    EngorgioFramePool *framepool;
    std::vector<EngorgioFrame*> frames;
    std::vector<EngorgioFrame*> sr_frames;
    int input_resolution, output_resolution, scale; // TODO: deprecated
    bool free_model;

    // inference
    int gpu_idx;
    nvinfer1::IExecutionContext* context = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    cudaStream_t cuda_stream = nullptr;
    std::vector<buffers_t> host_bindings;
    void *device_buffer[3] = {nullptr, nullptr, nullptr};

    // log 
    InferQueryLog log;

    QueryContext(int stream_id_, EngorgioModel *model_, EngorgioFramePool *framepool_, std::vector<EngorgioFrame*> &frames_, bool free_model_ = true)
    {
        free_model = free_model_;
        log.has_deadline = false;
        stream_id = stream_id_;
        model = model_;
        framepool = framepool_;
        frames = frames_;
        sr_frames.reserve(frames_.size());
        input_resolution = frames[0]->height;
        scale = model->scale_;
        output_resolution = input_resolution * scale;

        // TODO: copy frames
        // EngorgioFrame *frame;
        // while (frames_.size() > 0)
        // {
        //     frame = frames_.back();
        //     frames_.pop_back();
        //     frames.push_front(frame);
        // }

        context = nullptr;
        engine = nullptr;
        cuda_stream = nullptr;
        device_buffer[0] = nullptr;
        device_buffer[1] = nullptr;
        device_buffer[2] = nullptr;

        gpu_idx = 0; // TODO: remove this
    }

    QueryContext(int stream_id_, EngorgioModel *model_, EngorgioFramePool *framepool_, std::vector<EngorgioFrame*> &frames_, std::chrono::system_clock::time_point &deadline_, bool free_model_ = true)
    :  QueryContext(stream_id_, model_, framepool_, frames_, free_model_)
    {
        log.has_deadline = true;
        log.deadline = deadline_;
    }

    ~QueryContext()
    {
        // destroy model
        if (free_model && model)
            delete model;
        
        // destroy frames
        if (framepool)
        {
            for (unsigned int i = 0; i < frames.size(); i ++)
            {
                framepool->FreeFrame(frames[i]);
            }
        }
        else
        {
            for (unsigned int i = 0; i < frames.size(); i ++)
            {
                delete frames[i];
            }
        }   
    }
};

// TODO: merge dload and hload?
class InferEngine
{
template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

private:
    InferEngineType type_;
    unsigned int num_gpus_;
    std::vector<std::vector<int>> thread_indexes_;
    
    // engine
    int num_duplicates_;
    std::map<std::string, RefitEngine*> rengines_;

    // encoder
    JPEGEncodeEngine *jpeg_encode_engine_ = nullptr;
    libvpxEngine *libvpx_encode_engine_ = nullptr;

    // query
    std::vector<QueryContext*> queries_;
    std::deque<int> query_ids_;
    std::mutex query_mutex_;

    // cuda
    std::vector<std::deque<cudaStream_t>> cuda_streams_;
    std::mutex cuda_mutex_;
    cuDeviceMemory* device_memory_;
    cuHostMemoryV2* host_memory_;
    unsigned long long free_size_;

    // hhost thread
    // std::mutex hload_emutex_, hload_nmutex_;
    // std::deque<InferEvent> hload_events_;
    // std::vector<std::thread*> hload_workers_;
    // int num_hload_streams_;
    std::mutex hload_emutex_;
    std::vector<std::mutex*> hload_nmutexes_;
    std::deque<InferEvent> hload_events_;
    std::vector<std::thread*> hload_workers_;
    std::vector<int> num_hload_streams_;

    // infer thread
    // std::mutex infer_mutex_;
    // std::deque<InferEvent> infer_events_;
    // std::vector<std::thread*> infer_workers_;
    std::vector<std::mutex*> infer_mutexes_;
    std::vector<std::deque<InferEvent>> infer_events_;
    std::vector<std::thread*> infer_workers_;

    // unload thread
    // std::mutex unload_mutex_;
    // std::deque<InferEvent> unload_events_;
    // std::vector<std::thread*> unload_workers_;
    std::vector<std::mutex*> unload_mutexes_;
    std::vector<std::deque<InferEvent>> unload_events_;
    std::vector<std::thread*> unload_workers_;
    

    // log
    std::chrono::system_clock::time_point start_;
    std::vector<std::deque<InferQueryLog>> logs_;
    std::vector<std::mutex*> log_mutexes_;
    bool save_log_;
    std::filesystem::path log_dir_;

    void *DeviceMalloc(int gpu_idx, unsigned long long size);
    void DeviceFree(int gpu_idx, void *);
    void *HostMalloc(int resolution);
    void HostFree(int resolution, void *);
    bool LoadEngine(EngorgioModel *model, int gpu_idx);
    void SaveImage(std::vector<EngorgioFrame*> &frames);
public:
    InferEngine(InferEngineProfile &profile, bool mem_prealloc = true);
    InferEngine(InferEngineProfile &profile, JPEGEncodeEngine *jpeg_encode_engine_);
    InferEngine(InferEngineProfile &profile, libvpxEngine *libvpx_encode_engine_);
    ~InferEngine();

    bool Refit(int query_id);
    bool DeviceLoad(int query_id);
    bool HostLoad(int query_id);
    void Infer(int query_id, bool asynch);
    void Unload(int query_id);

    void HostLoadHandler(int gpu_idx);
    void InferHandler(int gpu_idx);
    void UnloadHandler(int gpu_idx);

    bool LoadEngine(EngorgioModel *model);
    // TODO: define and use query
    void EnhanceAsync(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, bool free_model = true); //TODO: replace uint8_t with yuv buffer
    void EnhanceAsync(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &dealine, bool free_model = true); //TODO: replace uint8_t with yuv buffer
    // void EnhanceSync(int stream_id, InferModel *model, std::vector<EngorgioFrame*> &frames, int input_resolution, int output_resolution); //TODO: replace uint8_t with yuv buffer
    bool Finished(); // for debugging
    void SaveLog();

    unsigned int GetGPUs();
};