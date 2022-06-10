#include <thread>
#include <chrono>
#include <set>
#include <cuda_runtime.h>
#include <nvml.h>
#include "parserOnnxConfig.h"
#include "logger.h"
#include "common.h"
#include "cudaUtility.h"
#include "date/date.h"
#include "enhancer_common.h"
#include "control_common.h"
#include "encode_engine.h"
#include "libvpx_engine.h"
#include "infer_engine.h"
#include "control_common.h"
#include "ipp.h"
#include "ippcc.h"
#include "ippi.h"
#include "enhancer.grpc.pb.h"

const int MAX_QURIES = 2000;

using namespace std;
namespace fs = std::filesystem;

template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

static unsigned long long get_free_size(int id)
{
    nvmlDevice_t device;
    nvmlMemory_t memory;

    // auto start = std::chrono::high_resolution_clock::now();

    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        throw std::runtime_error("1");
    }

    result = nvmlDeviceGetHandleByIndex(id, &device);
    if (NVML_SUCCESS != result)
    {
        throw std::runtime_error("2");
    }
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (NVML_SUCCESS != result)
    {
        throw std::runtime_error("3");
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (get_free_size): " << elapsed.count() * 1000 << "ms" << std::endl;

    return memory.free;
}

// TODO: print a subpixel convolution layer 
static std::pair<std::vector<std::string>, std::vector<nvinfer1::WeightsRole>> getLayerWeightsRolePair(nvinfer1::IRefitter& refitter)
{
    auto const num_All = refitter.getAll(0, nullptr, nullptr);

    // Allocate buffers for the items
    std::vector<const char*> layerNames(num_All);
    std::vector<nvinfer1::WeightsRole> weightsRoles(num_All);
    // Get
    refitter.getAll(num_All, layerNames.data(), weightsRoles.data());
    // container for std::string
    std::vector<std::string> layerNameStrs(num_All);
    
    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name)
    {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}

static std::vector<std::pair<nvinfer1::WeightsRole, nvinfer1::Weights>> getAllRefitWeightsForLayer(const nvinfer1::ILayer& l)
{
    switch (l.getType())
    {
    case nvinfer1::LayerType::kCONSTANT:
    {
        const auto& layer = static_cast<const nvinfer1::IConstantLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kCONSTANT, layer.getWeights())};
    }
    case nvinfer1::LayerType::kCONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IConvolutionLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kDECONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IDeconvolutionLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kFULLY_CONNECTED:
    {
        const auto& layer = static_cast<const nvinfer1::IFullyConnectedLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(nvinfer1::WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case nvinfer1::LayerType::kSCALE:
    {
        const auto& layer = static_cast<const nvinfer1::IScaleLayer&>(l);
        return {std::make_pair(nvinfer1::WeightsRole::kSCALE, layer.getScale()),
            std::make_pair(nvinfer1::WeightsRole::kSHIFT, layer.getShift())};
    }
    case nvinfer1::LayerType::kRNN_V2:
    case nvinfer1::LayerType::kACTIVATION:
    case nvinfer1::LayerType::kPOOLING:
    case nvinfer1::LayerType::kLRN:
    case nvinfer1::LayerType::kSOFTMAX:
    case nvinfer1::LayerType::kSHUFFLE:
    case nvinfer1::LayerType::kCONCATENATION:
    case nvinfer1::LayerType::kELEMENTWISE:
    case nvinfer1::LayerType::kPLUGIN:
    case nvinfer1::LayerType::kUNARY:
    case nvinfer1::LayerType::kPADDING:
    case nvinfer1::LayerType::kREDUCE:
    case nvinfer1::LayerType::kTOPK:
    case nvinfer1::LayerType::kGATHER:
    case nvinfer1::LayerType::kMATRIX_MULTIPLY:
    case nvinfer1::LayerType::kRAGGED_SOFTMAX:
    case nvinfer1::LayerType::kIDENTITY:
    case nvinfer1::LayerType::kPLUGIN_V2:
    case nvinfer1::LayerType::kSLICE:
    case nvinfer1::LayerType::kFILL:
    case nvinfer1::LayerType::kSHAPE:
    case nvinfer1::LayerType::kPARAMETRIC_RELU:
    case nvinfer1::LayerType::kRESIZE:
    case nvinfer1::LayerType::kTRIP_LIMIT:
    case nvinfer1::LayerType::kRECURRENCE:
    case nvinfer1::LayerType::kITERATOR:
    case nvinfer1::LayerType::kLOOP_OUTPUT:
    case nvinfer1::LayerType::kSELECT:
    default:
     return {};
    // case nvinfer1::LayerType::kQUANTIZE:
    // case nvinfer1::LayerType::kDEQUANTIZE: return {};
    }
    return {};
}

static std::pair<std::vector<std::string>, std::vector<nvinfer1::WeightsRole>> getMissingLayerWeightsRolePair(nvinfer1::IRefitter& refitter)
{
    auto const num_Missing = refitter.getMissing(0, nullptr, nullptr);

    // Allocate buffer for the items
    std::vector<const char*> layerNames(num_Missing);
    std::vector<nvinfer1::WeightsRole> weightsRoles(num_Missing);  
    // Get
    refitter.getMissing(num_Missing, layerNames.data(), weightsRoles.data());
    // container for std::string
    std::vector<std::string> layerNameStrs(num_Missing);

    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name)
    {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}

static bool RefitCudaEngine(nvinfer1::ICudaEngine *engine, nvinfer1::INetworkDefinition* network)
{
    // auto start = std::chrono::system_clock::now();

    auto const num_Layers = network->getNbLayers();
    // TODO: unique_ptr, glogger
    SampleUniquePtr<nvinfer1::IRefitter> refitter{nvinfer1::createInferRefitter(*engine, sample::gLogger.getTRTLogger())};
    auto const& layerWeightsRolePair = getLayerWeightsRolePair(*refitter);
    
    // used for checking existence
    std::set<std::pair<std::string, nvinfer1::WeightsRole>> layerRoleSet;
    auto const& layerNames = layerWeightsRolePair.first;
    auto const& weightsRoles = layerWeightsRolePair.second;

    // fill the set
    std::transform(layerNames.begin(), layerNames.end(), weightsRoles.begin(), 
        std::inserter(layerRoleSet, layerRoleSet.begin()),
        [](std::string const& layerName, nvinfer1::WeightsRole const& role) { return std::make_pair(layerName, role); });

    // function to check is refittable
    auto const isRefittable = [&layerRoleSet](char const* layerName, nvinfer1::WeightsRole const role) {
        return layerRoleSet.find(std::make_pair(layerName, role)) != layerRoleSet.end();
    };

    // function to refit the engine using network weights
    auto const setWeights = [&] {
        for (int32_t i=0; i < num_Layers; i++)
        {
            auto const layer = network->getLayer(i);
            auto const rolWeightsVec = getAllRefitWeightsForLayer(*layer);
            for (auto const& roleWeights : rolWeightsVec)
            {
                if (isRefittable(layer->getName(), roleWeights.first))
                {
                    bool const success = refitter->setWeights(layer->getName(), roleWeights.first, roleWeights.second);
                    if (!success)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto const reportMissingWeights = [&] {
        auto const& missingPair = getMissingLayerWeightsRolePair(*refitter);
        auto const& layerNames = missingPair.first;
        // auto const& weightsRoles = missingPair.second;
        for (size_t i = 0; i < layerNames.size(); i++)
        {
            sample::gLogError << "Missing (" << layerNames[i] << ") for refitting." << std::endl;
        }
        return layerNames.empty();
    };

    // Do refitting and check missing weights
    bool const success = setWeights() && reportMissingWeights() && refitter->refitCudaEngine();
    if (!success)
    {
        return false;
    }

    // auto end = std::chrono::system_clock::now();    
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (RefitCudaEngine): " << elapsed.count() * 1000 << "ms" << std::endl;

    return true;
}

static void SetupCudaStreams(int gpu_idx, std::vector<std::deque<cudaStream_t>> &cuda_streams)
{
    if (cudaSetDevice(gpu_idx) != cudaSuccess)
        throw std::runtime_error("GPU is not available");
    for (int i = 0; i < MAX_CUDA_STREAMS; i++)
    {
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess)
            throw std::runtime_error("Creating a CUDA stream failed");
        cuda_streams[gpu_idx].push_back(stream);
    }
}

EngorgioFrame* CopyFrame(EngorgioFrame *src_frame, int scale)
{
    EngorgioFrame *dest_frame = new EngorgioFrame();
    dest_frame->width = src_frame->width * scale;
    dest_frame->height = src_frame->height * scale;
    dest_frame->current_video_frame = src_frame->current_video_frame;
    dest_frame->current_super_frame = src_frame->current_super_frame;
    
    return dest_frame;
}


InferEngine::InferEngine(InferEngineProfile &profile, bool mem_prealloc)
{
    num_gpus_ = profile.num_gpus;
    type_ = profile.type;
    num_duplicates_ = NUM_DEPULICATES;
    queries_ = std::vector<QueryContext*>(MAX_QURIES, nullptr);
    for (std::size_t i = 0; i < queries_.size(); i++)
        query_ids_.push_back(i);
    save_log_ = profile.save_log;
    log_dir_ = profile.log_dir;
    start_ = profile.start;
    thread_indexes_ = profile.thread_indexes;
    
    std::vector<int> gpu_ids;
    for (std::size_t i = 0; i < num_gpus_; i++)
        gpu_ids.push_back(i);

    if (mem_prealloc && type_ == InferEngineType::kEngorgio)
    {
        device_memory_ = cuDeviceMemory::GetInstance(gpu_ids);
        host_memory_ = cuHostMemoryV2::GetInstance(num_gpus_);
        if (device_memory_ == nullptr or host_memory_ == nullptr)
        {
            cuDeviceMemory::RemoveInstance();
            cuHostMemoryV2::RemoveInstance();
            std::runtime_error("Memory pre-allocation failed");
        }
    }

    // cuda stream
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        cuda_streams_.emplace_back(std::deque<cudaStream_t>());
        SetupCudaStreams(i, cuda_streams_);

    }
    free_size_ = get_free_size(0);

    // log
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        logs_.emplace_back(std::deque<InferQueryLog>());
        log_mutexes_.push_back(new std::mutex());
    }

    // encode engine
    jpeg_encode_engine_ = nullptr;
    libvpx_encode_engine_ = nullptr;

    cpu_set_t cpuset;
    int rc;

    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        hload_nmutexes_.push_back(new std::mutex());
        num_hload_streams_.push_back(0);
        infer_events_.push_back(std::deque<InferEvent>());
        infer_mutexes_.push_back(new std::mutex());
        unload_events_.push_back(std::deque<InferEvent>());
        unload_mutexes_.push_back(new std::mutex());
    }

    // for (unsigned int i = 0; i < num_gpus_; i++)
    // {
    //     hload_workers_.push_back(new std::thread([this, i](){this->HostLoadHandler(i);}));
    //     CPU_ZERO(&cpuset);
    //     CPU_SET(thread_indexes_[i][0], &cpuset);
    //     rc = pthread_setaffinity_np(hload_workers_[i]->native_handle(),
    //                                  sizeof(cpu_set_t), &cpuset);
    //     assert (rc == 0);
    // }
    
        
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        infer_workers_.push_back(new std::thread([this, i](){this->InferHandler(i);}));
        CPU_ZERO(&cpuset);
        CPU_SET(thread_indexes_[i][0], &cpuset);
        rc = pthread_setaffinity_np(infer_workers_[i]->native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
        assert (rc == 0);
    }
        
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        unload_workers_.push_back(new std::thread([this, i](){this->UnloadHandler(i);}));
        CPU_ZERO(&cpuset);
        CPU_SET(thread_indexes_[i][1], &cpuset);
        rc = pthread_setaffinity_np(unload_workers_[i]->native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
        assert (rc == 0);
    }

}

InferEngine::InferEngine(InferEngineProfile &profile, JPEGEncodeEngine *jpeg_encode_engine): InferEngine(profile)
{
    // encode engine
    jpeg_encode_engine_ = jpeg_encode_engine;
}

InferEngine::InferEngine(InferEngineProfile &profile, libvpxEngine *libvpx_encode_engine): InferEngine(profile)
{
    // encode engine
    libvpx_encode_engine_ = libvpx_encode_engine;
}

InferEngine::~InferEngine()
{
    // logging
    if (save_log_)
        SaveLog();

    InferEvent ievent = {InferEventType::kJoin, 0};

    for (size_t i = 0; i < num_gpus_; i++)
    {
        hload_emutex_.lock();
        hload_events_.push_back(ievent);
        hload_emutex_.unlock();
    }
    for (size_t i = 0; i < num_gpus_; i++)
    {
    //     hload_workers_[i]->join();
    //     delete hload_workers_[i];
        delete hload_nmutexes_[i];
    }
    for (size_t i = 0; i < num_gpus_; i++)
    {
        infer_mutexes_[i]->lock();
        infer_events_[i].push_back(ievent);
        infer_mutexes_[i]->unlock();
    }
    for (size_t i = 0; i < num_gpus_; i++)
    {
        infer_workers_[i]->join();
        delete infer_workers_[i];
        delete infer_mutexes_[i];
    }
    for (size_t i = 0; i < num_gpus_; i++)
    {
        unload_mutexes_[i]->lock();
        unload_events_[i].push_back(ievent);
        unload_mutexes_[i]->unlock();
    }
    for (size_t i = 0; i < num_gpus_; i++)
    {
        unload_workers_[i]->join();
        delete unload_workers_[i];
        delete unload_mutexes_[i];
    }

    ICudaEngine *engine;
    for (auto & [name, rengine]: rengines_)
    {
        for (auto & cuda_engines : rengine->cuda_engines_per_gpu)
        {
            while (cuda_engines.size() > 0)
            {
                engine = cuda_engines.front();
                cuda_engines.pop_front();
                engine->destroy();
            }
        }
    }
    cuDeviceMemory::RemoveInstance();
    cuHostMemoryV2::RemoveInstance();
}

// TODO: start from here
bool InferEngine::LoadEngine(EngorgioModel *model)
{
    rengines_[model->name_] = new RefitEngine(num_gpus_);
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        if (!LoadEngine(model, i))
        {
            delete model;
            return false;
        }
    }
    delete model;
    return true;
}

bool InferEngine::LoadEngine(EngorgioModel *model, int gpu_idx)
{
    // std::cout << "LoadEngine: " << gpu_idx << std::endl;

    if (cudaSetDevice(gpu_idx) != cudaSuccess)
        throw std::runtime_error("GPU is not available");

    auto item = rengines_.find(model->name_);
    if (rengines_.find(model->name_) == rengines_.end())
    {        
        std::cerr << "Model does not exist" << std::endl;
        return false;
    }
   
    RefitEngine *rengine = item->second;
    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        return false;
    }
    ICudaEngine *engine;
    for (int i = 0; i < num_duplicates_; i++)
    {
        engine = runtime->deserializeCudaEngine(model->buf_, model->size_);
        if (!engine)
        {
            std::cerr << "deserializeCudaEngine() failed" << std::endl;
            while (rengine->cuda_engines_per_gpu[gpu_idx].size() > 0)
            {
                engine = rengine->cuda_engines_per_gpu[gpu_idx].front();
                rengine->cuda_engines_per_gpu[gpu_idx].pop_front();
            }
            return false;
        }   
        rengine->cuda_engines_per_gpu[gpu_idx].push_back(engine);
    }

    return true;
}

void* InferEngine::DeviceMalloc(int gpu_idx, unsigned long long size)
{
    void *buf;
    unsigned long long curr_free_size;
    switch (type_)
    {
    case InferEngineType::kEngorgio:
        if (device_memory_)
            return device_memory_->Malloc(gpu_idx, size);
        else
            return nullptr;
        break;
    case InferEngineType::kBaseline1:
    case InferEngineType::kBaseline2:
        throw std::runtime_error("Not support multi-GPUs yet");
        curr_free_size = get_free_size(0);
        // if ((curr_free_size - size) > (0.1 * free_size_))
        if (size <= (curr_free_size - (1 - DEVICE_MEM_FRAC) * free_size_))
        {
            if (cudaMalloc(&buf, size) == cudaSuccess)
                return buf;
            else
                return nullptr;
        }
        else
            return nullptr;
        break;
    default:
        throw std::runtime_error("Unsupported inference engine type");
        break;
    }
}

void InferEngine::DeviceFree(int gpu_idx, void *buf)
{
    if (!buf)
        return;

    switch (type_)
    {
    case InferEngineType::kEngorgio:
        device_memory_->Free(gpu_idx, buf);
        break;
    case InferEngineType::kBaseline1:
    case InferEngineType::kBaseline2:
        throw std::runtime_error("Not support multi-GPUs yet");
        cudaFree(buf);
        break;
    default:
        throw std::runtime_error("Unsupported inference engine type");
        break;
    }
}

void* InferEngine::HostMalloc(int resolution) 
{
    size_t size;
    switch (type_)
    {
    case InferEngineType::kEngorgio:
        if (host_memory_)
            return host_memory_->Malloc(resolution);
        else
            return nullptr;
        break;
    case InferEngineType::kBaseline1:
        void *buf;
        size = get_frame_size(resolution);
        // return (void*) malloc(size);
        // std::cout << size << std::endl;
        if(cudaHostAlloc(&buf, size, cudaHostAllocDefault) == cudaSuccess)
            return buf;
        else
            return nullptr;
        break;
    case InferEngineType::kBaseline2:
        size = get_frame_size(resolution);
        return (void*) malloc(size);
        break;
    default:
        throw std::runtime_error("Unsupported inference engine type");
        break;
    }
}

void InferEngine::HostFree(int resolution, void *buf)
{
    if (!buf)
        return;
    
    switch (type_)
    {
    case InferEngineType::kEngorgio:
        host_memory_->Free(resolution, buf);
        break;
    case InferEngineType::kBaseline1:
        // free(buf);
        cudaFreeHost(buf);
        break;
    case InferEngineType::kBaseline2:
        free(buf);
        break;
    default:
        throw std::runtime_error("Unsupported inference engine type");
        break;
    }
}

void InferEngine::HostLoadHandler(int gpu_idx)
{
    InferEvent ievent;
    bool has_event = false;

    if (cudaSetDevice(gpu_idx) != cudaSuccess)
        throw std::runtime_error("GPU is not available");

    while (1)
    {
        hload_emutex_.lock();
        has_event = false;

        if (!hload_events_.empty())
        {
            ievent = hload_events_.front();
            hload_events_.pop_front();
            has_event = true;
        }
        hload_emutex_.unlock();

        if (has_event)
        {
            // std::cout << "hostload event" << std::endl;
            switch (ievent.type)
            {
            case InferEventType::kQuery:
                queries_[ievent.query_id]->gpu_idx = gpu_idx;
                if (HostLoad(ievent.query_id))
                {                    
                    // send an infer event
                    infer_mutexes_[gpu_idx]->lock();
                    infer_events_[gpu_idx].push_back(ievent);
                    infer_mutexes_[gpu_idx]->unlock();
                }
                else
                {
                    // send an unload event
                    unload_mutexes_[gpu_idx]->lock();
                    unload_events_[gpu_idx].push_back(ievent);
                    unload_mutexes_[gpu_idx]->unlock();
                }
                break;
            case InferEventType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}

void InferEngine::InferHandler(int gpu_idx)
{
    InferEvent ievent;
    bool has_event = false;
    // std::cout << "InferHandler: " << gpu_idx << std::endl;

    if (cudaSetDevice(gpu_idx) != cudaSuccess)
        throw std::runtime_error("GPU is not available");

    while (1)
    {
        // infer_mutexes_[gpu_idx]->lock();
        // has_event = false;
        // if (!infer_events_[gpu_idx].empty())
        // {
        //     ievent = infer_events_[gpu_idx].front();
        //     infer_events_[gpu_idx].pop_front();
        //     has_event = true;
        // }
        // infer_mutexes_[gpu_idx]->unlock();
        hload_emutex_.lock();
        has_event = false;

        if (!hload_events_.empty())
        {
            ievent = hload_events_.front();
            hload_events_.pop_front();
            has_event = true;
        }
        hload_emutex_.unlock();

        if (has_event)
        {
            // std::cout << "infer event" << std::endl;
            switch (ievent.type)
            {
            case InferEventType::kQuery:
                queries_[ievent.query_id]->gpu_idx = gpu_idx;
                if (HostLoad(ievent.query_id))
                    Infer(ievent.query_id, true);
                unload_mutexes_[gpu_idx]->lock();
                unload_events_[gpu_idx].push_back(ievent);
                unload_mutexes_[gpu_idx]->unlock();
                break;
            case InferEventType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}

void InferEngine::UnloadHandler(int gpu_idx)
{
    InferEvent ievent;
    bool has_event = false;

    if (cudaSetDevice(gpu_idx) != cudaSuccess)
        throw std::runtime_error("GPU is not available");

    while (1)
    {
        unload_mutexes_[gpu_idx]->lock();
        has_event = false;
        if (!unload_events_[gpu_idx].empty())
        {
            ievent = unload_events_[gpu_idx].front();
            unload_events_[gpu_idx].pop_front();
            has_event = true;
        }
        unload_mutexes_[gpu_idx]->unlock();

        if (has_event)
        {
            // std::cout << "unload event" << std::endl;
            switch (ievent.type)
            {
            case InferEventType::kQuery:
                Unload(ievent.query_id);
                break;
            case InferEventType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}
bool InferEngine::Refit(int query_id)
{
    // std::cout << "Refit Thread is running on CPU " << sched_getcpu() << "\n";
    QueryContext *query = queries_[query_id];
    int gpu_idx = query->gpu_idx;
    RefitEngine *rengine = nullptr;
    ICudaEngine *engine;
    auto item = rengines_.find(query->model->name_);
    if (item == rengines_.end())
    {
        return false;
        
    }
    rengine = item->second; 
    // TODO: change the follows in a thread-safe way 
    bool found = false;
    while(!found)
    {
        rengine->mutexes_per_gpu[gpu_idx]->lock();
        // std::cout << "size: " << rengines->cuda_engines.size() << std::endl;
        if (rengine->cuda_engines_per_gpu[gpu_idx].size() > 0)
        {
            engine = rengine->cuda_engines_per_gpu[gpu_idx].front();
            rengine->cuda_engines_per_gpu[gpu_idx].pop_front();
            query->engine = engine;
            found = true;
        }
        rengine->mutexes_per_gpu[gpu_idx]->unlock();
    }
    
    auto start = std::chrono::system_clock::now();

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
        return false;
	}

    auto parsed = parser->parse(query->model->buf_, query->model->size_);  
    if (!parsed)
    {
        return false;
    }

    if (!RefitCudaEngine(query->engine, network.get()))
    {
        std::cerr << "Refit failed" << std::endl;
        return false;
    }

    nvinfer1::IExecutionContext* context = query->engine->createExecutionContextWithoutDeviceMemory();
    query->context = context;
    if (!context)
    {
        std::cerr << "CreateContext failed" << std::endl;
        return false;
    }

    auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Refit): " << elapsed.count() << "ms" << std::endl;

    if (save_log_)
    {
        query->log.refit_start = start;
        query->log.refit_end = end;
    }

    return true;
}

bool InferEngine::DeviceLoad(int query_id)
{
    // std::cout << "DeviceLoad Thread is running on CPU " << sched_getcpu() << "\n";

    QueryContext *query = queries_[query_id];
    int gpu_idx = query->gpu_idx;
    nvinfer1::IExecutionContext* context = query->context;
    nvinfer1::ICudaEngine* engine = query->engine;
    
    int input_idx = engine->getBindingIndex("input");
    int output_idx = engine->getBindingIndex("output");
    nvinfer1::Dims input_dims = engine->getBindingDimensions(input_idx);
    nvinfer1::Dims output_dims = engine->getBindingDimensions(output_idx);  
    size_t input_size = input_dims.d[1] * input_dims.d[2] * input_dims.d[3];
    size_t output_size = output_dims.d[1] * output_dims.d[2] * output_dims.d[3];
    // int bytes_per_pixel = 2;
    int bytes_per_pixel = 4;
    int batch_size = 1;

    // std::cout << input_dims.d[0] << "," << input_dims.d[1] << "," << input_dims.d[2] << "," << input_dims.d[3] <<std::endl;
    // std::cout << output_dims.d[0] << "," << output_dims.d[1] << "," << output_dims.d[2] << "," << output_dims.d[3] <<std::endl;

    size_t load_size = 0;
    size_t aligned = 256;
    size_t input_aligned, output_aligned, inter_aligned;
    load_size += input_size * batch_size * bytes_per_pixel;
    input_aligned = (load_size % aligned);
    load_size += input_aligned;
    load_size += output_size * batch_size * bytes_per_pixel;
    output_aligned = (load_size % aligned);
    load_size += output_aligned;
    load_size += output_size * batch_size * 1;
    inter_aligned = (load_size % aligned);
    load_size += inter_aligned;
    load_size += engine->getDeviceMemorySize();



    while ((query->device_buffer[0] = DeviceMalloc(gpu_idx, load_size)) == nullptr)
    {
    }
    auto start = std::chrono::system_clock::now();
    query->device_buffer[1] = (uint8_t*)query->device_buffer[0] + input_size * batch_size * bytes_per_pixel + input_aligned;
    query->device_buffer[2] = (uint8_t*)query->device_buffer[1] + output_size * batch_size * bytes_per_pixel + output_aligned;
    context->setDeviceMemory((uint8_t*)query->device_buffer[2] + output_size * batch_size + inter_aligned); 

    auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Device memory): " << elapsed.count() << "ms" << std::endl;

    if (save_log_)
    {
        query->log.device_start = start;
        query->log.device_end = end;
    }

    return true;
}

bool InferEngine::HostLoad(int query_id)
{
    QueryContext *query = queries_[query_id];
    bool found = false;
    int gpu_idx = query->gpu_idx;
    while (!found)
    {
        hload_nmutexes_[gpu_idx]->lock();
        if (num_hload_streams_[gpu_idx] < MAX_HLOAD_STREAMS)
        {
            found = true;
            num_hload_streams_[gpu_idx] += 1;
        }
        hload_nmutexes_[gpu_idx]->unlock();
    }   
    // while (!found)
    // {
    //     hload_nmutexes_[gpu_idx]->lock();
    //     if (num_hload_streams_[gpu_idx] < MAX_HLOAD_STREAMS)
    //     {
    //         found = true;
    //         num_hload_streams_[gpu_idx] += 1;
    //     }
    //     hload_nmutexes_[gpu_idx]->unlock();
    // }   

    auto start = std::chrono::system_clock::now();

    // host memory allocation
    buffers_t host_binding;
    std::vector<buffers_t> &host_bindings = query->host_bindings;
    host_bindings.reserve(query->frames.size()); // TODO: same effect as reserve?
    bool success = true;

    size_t input_size = get_frame_size(query->input_resolution);
    EngorgioFrame *sr_frame;
    // IppiSize roi_size;
    // IppStatus st = ippStsNoErr;
    // Ipp8u* const rgb_buffers[3] = {&lr_rgb_buffers_[gpu_idx][0], &lr_rgb_buffers_[gpu_idx][input_size / 3],  &lr_rgb_buffers_[gpu_idx][2 * input_size / 3]};

    for (size_t i = 0; i < query->frames.size(); i++)
    {
            // if ((host_binding.first = host_memory_->Malloc(input_size * levent.batch_size)) == nullptr);
        if ((host_binding.first = HostMalloc(query->input_resolution)) == nullptr)
        {
            std::cerr << "Host input malloc failed" << std::endl;
            success = false;
            break;
        }
        if ((host_binding.second = HostMalloc(query->output_resolution)) == nullptr)
        {
            // std::cout << "Output size: " << event.output_size << std::endl;
            std::cerr << "Host output malloc failed" << std::endl;
            success = false;
            break;
        }
        
        // auto start3 = std::chrono::high_resolution_clock::now();
        // roi_size.width = query->frames[i]->width;
        // roi_size.height = query->frames[i]->height;
        // st = ippiCopy_8u_C3P3R((const Ipp8u*) query->frames[i]->rgb_buffer, query->frames[i]->width * 3, rgb_buffers, query->frames[i]->width, roi_size);
        // if ( st != ippStsNoErr) 
        //     throw std::runtime_error("ippiCopy_8u_C3P3R() failed");
        // auto end3 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed3 = end3 - start3;
        // std::cout << elapsed3.count() * 1000 << std::endl;

        if (cudaMemcpy(host_binding.first, query->frames[i]->rgb_buffer, input_size, cudaMemcpyHostToHost) != cudaSuccess)
        {
            std::cerr << "Host memcpy has failed" << std::endl;
            success = false;
            break;
        }

        host_bindings.push_back(host_binding);
        sr_frame = CopyFrame(query->frames[i], query->scale);
        query->sr_frames.push_back(sr_frame);
    }

    auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Host): " << elapsed.count() << "ms" << std::endl;

    if (save_log_)
    {
        query->log.host_start = start;
        query->log.host_end = end;
    }

    return success;
}

void InferEngine::Infer(int query_id, bool asynch)
{
    // std::cout << "Infer Thread is running on CPU " << sched_getcpu() << "\n";
    if (!Refit(query_id) || !DeviceLoad(query_id))
    {
        std::cerr << "Reift() or Deviceload() failed" << std::endl;
        return;
    }

    QueryContext *query = queries_[query_id];
    int gpu_idx = query->gpu_idx;
    bool found = false;
    while (!found)
    {
        cuda_mutex_.lock();
        if (cuda_streams_[gpu_idx].size() > 0)
        {
            query->cuda_stream = cuda_streams_[gpu_idx].front();
            cuda_streams_[gpu_idx].pop_front();
            found = true;
        }
        cuda_mutex_.unlock();
    }

    auto start = std::chrono::system_clock::now();

    size_t input_size = get_frame_size(query->input_resolution);
    size_t output_size = get_frame_size(query->output_resolution);
    for (auto& host_binding : query->host_bindings)
    {
        if (cudaMemcpyAsync(query->device_buffer[2], host_binding.first, input_size, cudaMemcpyHostToDevice, query->cuda_stream) != cudaSuccess)
        {
            std::cerr << "cudaMemcpyAsynch (host->device) failed" << std::endl;
            break;
        }
        // uchar2halfArray((uint8_t*)query->device_buffer[2], (uint16_t*)query->device_buffer[0], input_size, query->cuda_stream);
        uchar2floatArray((uint8_t*)query->device_buffer[2], (uint32_t*)query->device_buffer[0], input_size, query->cuda_stream);

        if (!query->context->enqueue(1, query->device_buffer, query->cuda_stream, nullptr))
        {
            std::cerr << "enqueue failed" << std::endl;
            break;
        }

        // half2ucharArray((uint16_t*)query->device_buffer[1], (uint8_t*)query->device_buffer[2], output_size, query->cuda_stream);
        float2ucharArray((uint32_t*)query->device_buffer[1], (uint8_t*)query->device_buffer[2], output_size, query->cuda_stream);
        if (cudaMemcpyAsync(host_binding.second, query->device_buffer[2], output_size, cudaMemcpyDeviceToHost,  query->cuda_stream) != cudaSuccess)
        {
            std::cerr << "cudaMemcpyAsynch (device -> host) failed" << std::endl;
            break;
        }
    }    

    // cudaStreamSynchronize(query->cuda_stream); // TODO: remove this

    if(!asynch) 
    {
        cudaStreamSynchronize(query->cuda_stream);
    }

    auto end = std::chrono::system_clock::now();

    if (save_log_)
    {
        query->log.infer_start = start;
        query->log.infer_end = end;
    }
}

void InferEngine::SaveImage(std::vector<EngorgioFrame*> &frames)
{
    std::string file_name;
    std::filesystem::path file_path;

    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        std::string today = date::format("%F", std::chrono::system_clock::now());   
        // log_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "infer_engine" / today;
        log_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "infer_engine";
    }
    if (!fs::exists(log_dir_))
        fs::create_directories(log_dir_);

    std::ofstream fout;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        file_name = std::to_string(i) + "_sr.rgb";
        file_path = log_dir_ / file_name;
        fout.open(file_path, std::ios::out | std::ios::binary);
        if (fout.is_open()){
            fout.write((const char*)frames[i]->rgb_buffer, frames[i]->width * frames[i]->height * 3);
            fout.close();
        }

        // uint8_t *hr_img = (uint8_t*) host_bindings[i].second;
        // for (int j = 0 ; j < 1000; j++)
        //     std::cout << (int) hr_img[j] << std::endl;
        // break;
    }
}

void InferEngine::Unload(int query_id)
{
    // std::cout << "UnLoad Thread is running on CPU " << sched_getcpu() << "\n";
    QueryContext* query = queries_[query_id];
    int gpu_idx = query->gpu_idx;

    auto start = std::chrono::system_clock::now();

    // return cuda stream
    if (query->cuda_stream)
    {
        if (cudaStreamSynchronize(query->cuda_stream) != cudaSuccess)
        {
            std::cerr << "cudaStreamSynchronize failed" << std::endl;
        }
        cuda_mutex_.lock();
        cuda_streams_[gpu_idx].push_back(query->cuda_stream);
        cuda_mutex_.unlock();
    }

    // map sr frames
    for (std::size_t  i = 0; i < query->host_bindings.size(); i++)
    {
        query->sr_frames[i]->rgb_buffer = (uint8_t*) query->host_bindings[i].second;
    }

    // send an encode event or destroy a query
    switch (type_){
        case InferEngineType::kEngorgio:
            if (jpeg_encode_engine_)
            {
                jpeg_encode_engine_->Encode(query->stream_id, query->sr_frames);
                // jpeg_encode_engine_->Encode_kakadu(query->stream_id, query->sr_frames);
            }
            else if (libvpx_encode_engine_)
            {
                libvpx_encode_engine_->Encode(query->stream_id, query->sr_frames);
            }
            else
            {
                for (std::size_t  i = 0; i < query->host_bindings.size(); i++)
                {
                    HostFree(query->output_resolution, query->host_bindings[i].second);
                    query->sr_frames[i]->rgb_buffer = nullptr;
                    delete query->sr_frames[i];
                }
            }
            break;
        default:
            std::runtime_error("Not implemented");
            break;
    }

    // return cuda engines
    if (query->engine)
    {
        auto item = rengines_.find(query->model->name_);
        RefitEngine *rengine = item->second;
        rengine->mutexes_per_gpu[gpu_idx]->lock();
        rengine->cuda_engines_per_gpu[gpu_idx].push_back(query->engine);
        rengine->mutexes_per_gpu[gpu_idx]->unlock();
    }

    // destroy execution context
    if (query->context)
        query->context->destroy();

    // return host memory
    for (auto binding : query->host_bindings)
    {
        HostFree(query->input_resolution, binding.first);
    }
    hload_nmutexes_[gpu_idx]->lock();
    num_hload_streams_[gpu_idx] -= 1;
    hload_nmutexes_[gpu_idx]->unlock();
    
    // return device memory
    if (query->device_buffer[0])
        DeviceFree(gpu_idx, query->device_buffer[0]);

    auto end = std::chrono::system_clock::now();

            // auto start3 = std::chrono::high_resolution_clock::now();
        // roi_size.width = query->frames[i]->width;
        // roi_size.height = query->frames[i]->height;
        // st = ippiCopy_8u_C3P3R((const Ipp8u*) query->frames[i]->rgb_buffer, query->frames[i]->width * 3, rgb_buffers, query->frames[i]->width, roi_size);
        // if ( st != ippStsNoErr) 
        //     throw std::runtime_error("ippiCopy_8u_C3P3R() failed");
        // auto end3 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed3 = end3 - start3;
        // std::cout << elapsed3.count() * 1000 << std::endl;

    // save a log
    if (save_log_)
    {
        queries_[query_id]->log.unload_start = start;
        queries_[query_id]->log.unload_end = end;
        log_mutexes_[gpu_idx]->lock();
        logs_[gpu_idx].push_back(queries_[query_id]->log);
        log_mutexes_[gpu_idx]->unlock();
    }

    // destroy a model, a query
    delete query;    
    queries_[query_id] = nullptr;
    query_mutex_.lock();
    query_ids_.push_back(query_id);
    query_mutex_.unlock();
}

void InferEngine::EnhanceAsync(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, bool free_model)
{
    // allocate a query id
    QueryContext *new_query = new QueryContext(stream_id, model, framepool, frames, free_model);
    query_mutex_.lock();
    int query_id = query_ids_.front();
    query_ids_.pop_front();
    query_mutex_.unlock();
    queries_[query_id] = new_query;

    // logging 
    if (save_log_)
    {
        new_query->log.stream_id = stream_id;
        for (auto frame : frames)
        {
            new_query->log.video_indexes.push_back(frame->current_video_frame);
            new_query->log.super_indexes.push_back(frame->current_super_frame);
        }
    }
    
    // send a host load event
    InferEvent ievent = {InferEventType::kQuery, query_id};
    hload_emutex_.lock();
    hload_events_.push_back(ievent);
    hload_emutex_.unlock();
}

void InferEngine::EnhanceAsync(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &dealine, bool free_model)
{
    // std::cout << stream_id << "," << model->name_ << "," << frames.size() << std::endl;
    // return;

    // allocate a query id
    QueryContext *new_query = new QueryContext(stream_id, model, framepool, frames, dealine, free_model);
    query_mutex_.lock();
    int query_id = query_ids_.front();
    query_ids_.pop_front();
    query_mutex_.unlock();
    queries_[query_id] = new_query;

    // logging 
    if (save_log_)
    {
        new_query->log.stream_id = stream_id;
        for (auto & frame : frames)
        {
            new_query->log.video_indexes.push_back(frame->current_video_frame);
            new_query->log.super_indexes.push_back(frame->current_super_frame);
        }
    }

    // send a host load event
    InferEvent ievent = {InferEventType::kQuery, query_id};
    hload_emutex_.lock();
    hload_events_.push_back(ievent);
    hload_emutex_.unlock();
}


// TODO: implement this 
// void InferEngine::EnhanceSync(int stream_id, InferModel *model, InferFrames *frames, int input_resolution, int output_resolution)
// {
//      // allocate a query id
//     QueryContext *new_query = new QueryContext(model->name_);
//     query_mutex_.lock();
//     int query_id = 0;
//     for (int i = 0; i < queries_.size(); i++)
//     {
//         if (queries_[i] == nullptr)
//         {
//             query_id = i;
//             queries_[i] = new_query;
//             break;
//         }
//     }
//     query_mutex_.unlock();

//     InferLog* log;
//     if(save_log_)
//     {
//         log = new InferLog;
//         new_query->log = log;
//     }

//     // send a deserializ event
//     RefitEvent revent;
//     revent.type = LoadEventType::kQuery;
//     revent.stream_id = stream_id;
//     revent.query_id = query_id;
//     revent.model = model;
//     Refit(revent);

//     // // send a host load event
//     HostLoadEvent hlevent;
//     hlevent.type = LoadEventType::kQuery;
//     hlevent.stream_id = stream_id;
//     hlevent.query_id = query_id;
//     hlevent.frames = frames;
//     hlevent.input_resolution = input_resolution;
//     hlevent.output_resolution = output_resolution;
//     HostLoad(hlevent);

//     // send a device load event 
//     DeviceLoadEvent dlevent;
//     dlevent.type = LoadEventType::kQuery;
//     dlevent.stream_id = stream_id;
//     dlevent.query_id = query_id;
//     dlevent.input_resolution = input_resolution;
//     dlevent.output_resolution = output_resolution;
//     DeviceLoad(dlevent, false);

//     // send an infer event
//     InferEvent ievent;
//     ievent.type = LoadEventType::kQuery;
//     ievent.stream_id = stream_id;
//     ievent.query_id = query_id;
//     ievent.input_resolution = input_resolution;
//     ievent.output_resolution = output_resolution;
//     Infer(ievent, false);

//     // send an unload event 
//     UnloadEvent uevent;
//     uevent.type = LoadEventType::kQuery;
//     uevent.stream_id = stream_id;
//     uevent.query_id = query_id;
//     uevent.input_resolution = input_resolution;
//     uevent.output_resolution = output_resolution;
//     Unload(uevent);
// }

bool InferEngine::Finished()
{

    // query_mutex_.lock();
    // for (int i = 0; i < queries_.size(); i++)
    // {
    //     if (queries_[i] != nullptr)
    //     {
    //         finished = false;
    //         break;
    //     }
    // }
    // query_mutex_.unlock()
    // std::cout << query_ids_.size() << "," << queries_.size() << std::endl;

    if(query_ids_.size() != queries_.size())
        return false;
    else
        return true;
}

struct less_than_key
{
    inline bool operator() (const InferFrameLog *log1, const InferFrameLog *log2)
    {
        if (log1->video_index != log2->video_index)
            return (log1->video_index < log2->video_index);
        else
            return (log1->super_index < log2->super_index);
    }
};

void InferEngine::SaveLog()
{
    if (!save_log_)
        return;

    std::map<int, std::vector<InferFrameLog*>> frame_logs;

    // Build frame-level logs 
    InferFrameLog *frame_log;
    // std::cout << "query_log: " << logs_[0].size() << std::endl;
    for (unsigned int i = 0; i < num_gpus_; i++)
    {
        for (auto query_log : logs_[i])
        {
            int stream_id = query_log.stream_id;
            if (frame_logs.find(stream_id) == frame_logs.end())
            {
                frame_logs[stream_id] = std::vector<InferFrameLog*>();
            }
            for (std::size_t j = 0; j < query_log.video_indexes.size(); j++)
            {
                // query_log.video_indexes[j] = j;
                frame_log = new InferFrameLog(query_log, j);
                frame_logs[stream_id].push_back(frame_log);
            }
        }
    }

    // Sort frame-level logs
    for (auto & [id, logs] : frame_logs)
    {
        std::sort(logs.begin(), logs.end(), less_than_key());
    }

    // Save frame-level logs 
    if (!save_log_)
        return;

    // set base dir and create it
    std::filesystem::path base_log_dir, log_dir; 
    std::string today = date::format("%F", std::chrono::system_clock::now());   
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
        // base_log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / today;
        base_log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results";
    }
    else
    {
        base_log_dir = log_dir_;
        // base_log_dir = log_dir_ / today;
    }
      
    // anchor index log
    fs::path log_path;
    std::ofstream log_file;
    std::chrono::duration<double> start_elapsed, end_elapsed;

    for (auto & [id, logs] : frame_logs)
    {
        log_dir = base_log_dir / std::to_string(id);
        if (!fs::exists(log_dir))
            fs::create_directories(log_dir);

        log_path = log_dir / "infer_latency.txt";
        log_file.open(log_path);
        if (log_file.is_open())
        {
            log_file << "Video index\tSuper index\t" 
                    <<  "Host (start)\tHost (end)\t"
                    <<  "Refit (start)\tRefit (end)\t"
                    <<  "Device (start)\tDevice (end)\t"
                    <<  "Infer (start)\tInfer (end)\t"
                    <<  "Unload (start)\tUnload (end)\t"
                    <<  "Deadline\t"
                    <<  "\n";
            for (auto log : logs)
            {
                log_file << std::to_string(log->video_index) << "\t"
                         << std::to_string(log->super_index) << "\t";

                start_elapsed = log->host_start - start_;
                end_elapsed = log->host_end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";

                start_elapsed = log->refit_start - start_;
                end_elapsed = log->refit_end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";

                start_elapsed = log->device_start - start_;
                end_elapsed = log->device_end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";

                start_elapsed = log->infer_start - start_;
                end_elapsed = log->infer_end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";

                start_elapsed = log->unload_start - start_;
                end_elapsed = log->unload_end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";

                if (log->has_deadline)
                {
                    start_elapsed = log->deadline - start_;
                    log_file << std::to_string(start_elapsed.count()) << "\t";
                }

                log_file << "\n";
            }
        }
        log_file.flush();
        log_file.close();
    }

    // Free frame-level logs
    for (auto & [id, logs] : frame_logs)
    {
        for (auto log: logs)
            delete log;
    }
}

// void InferEngine::SaveLog()
// {
//     // set base dir and create it 
//     if (save_dir_.empty())
//     {
//         // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
//         std::string today = date::format("%F", std::chrono::system_clock::now());   
//         save_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "infer_engine" / today;
//     }
//     if (!fs::exists(save_dir_))
//         fs::create_directories(save_dir_);
       
//     // save logs
//     fs::path index_path;
//     std::ofstream index_file;
//     std::string index_log;

//     for (int i = 0; i < num_gpus_; i++)
//     {
//         index_path = save_dir_ / ("latency" + std::to_string(int(type_)) + "_gpu" + std::to_string(i) + ".txt");
//         index_file.open(index_path);
//         if (index_file.is_open())
//         {
//             index_file << "refit 1\trefit 2\thload 1\thload 2\tdload\tinfer1\tinter2\tunload1\tunload2\n";
        
//             for (auto log : logs_[i])
//             {
//                 index_file << std::to_string(log.refit_latencies[0]) << "\t";
//                 index_file << std::to_string(log.refit_latencies[1]) << "\t";
//                 index_file << std::to_string(log.hload_latencies[0]) << "\t";
//                 index_file << std::to_string(log.hload_latencies[1]) << "\t";
//                 index_file << std::to_string(log.dload_latency) << "\t";
//                 index_file << std::to_string(log.infer_latency[0]) << "\t";
//                 index_file << std::to_string(log.infer_latency[1]) << "\t";
//                 index_file << std::to_string(log.unload_latencies[0]) << "\t";
//                 index_file << std::to_string(log.unload_latencies[1]) << "\n";
//             }
//         }
//         index_file.close();   
//     }
// } 

unsigned int InferEngine::GetGPUs()
{
    return num_gpus_;
}