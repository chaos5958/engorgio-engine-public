#include "BaseModel.h"
#include "common.h"
#include "logging.h"
#include "cudaUtility.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cassert>
#include <vector>

common::Logger BaseModel::shared_logger_; 

BaseModel::BaseModel(ModelType& model_type, nvinfer1::ILogger::Severity severity)
{
    model_type_ = model_type;
    logger_ = &shared_logger_;
    device_memory_ = cuDeviceMemory::GetInstance();

    if (model_type.mtype == MemoryType::kPreallocate && device_memory_ == nullptr)
        throw std::runtime_error("Device memory is not allocated yet");
    if (cudaSetDevice(model_type_.device_id) != cudaSuccess) // TODO: is this correct to set device here?
        throw std::runtime_error("GPU is not available");
    if (cudaStreamCreate(&stream_) != cudaSuccess)
        throw std::runtime_error("Creating a CUDA stream failed");
}

//TODO: check all pointers are deallocataed here correctly
BaseModel::~BaseModel()
{
    if (stream_)
        cudaStreamDestroy(stream_);
    if (context_)
        context_->destroy();
    FreeDeviceMemory();
    DestroyEngine();
}

bool BaseModel::BuildEngine(const std::string& file)
{
    if (is_built_)
    {
        common::LOG_INFO(*logger_) << "TRT engine is already built" << std::endl;
        return false;
    }

    switch (model_type_.btype)
    {
    case BuildType::kTRT:
        if (!BuildFromTRT(file))
            return false;
        break;
    case BuildType::kONNX:
        if (!BuildFromONNX(file))
            return false;
        break;
    case BuildType::kPyTorch:
        /*if (!BuildFromPyTorch(file))
            return false;*/
        break;
    default:
        common::LOG_ERROR(*logger_) << "Invalid type to build a TRT engine" << std::endl;
        break;
    }

    workspace_size_ = engine_->getDeviceMemorySize();
    is_built_ = true;

    return true;
}

void BaseModel::DestroyEngine()
{
    if (!is_built_)
        return;
    if (engine_)
        engine_->destroy();
    is_built_ = false;
}

size_t BaseModel::ReadFile(const std::string& file, void** buffer)
{
    size_t size;

    std::ifstream stream(file, std::ifstream::ate | std::ifstream::binary);
    if (!stream.is_open())
    {
        common::LOG_ERROR(*logger_) << "Cannot read a file: " << file << std::endl;
        return 0;
    }

    size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    *buffer = (unsigned char*)malloc(size);
    if (*buffer == nullptr)
        return 0;

    stream.read((char*)*buffer, size);
    return size;
}

bool BaseModel::BuildFromTRT(const std::string& file)
{
    void* buffer{ nullptr };
    size_t size = ReadFile(file, &buffer);
    
    if (size == 0)
        return false;

    auto runtime = UniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
    if (!runtime)
    {
        common::LOG_ERROR(*logger_) << "Creating the infer runtime failed" << std::endl;
        return false;
    }

    engine_ = runtime->deserializeCudaEngine(buffer, size);
    if (!engine_)
    {
        common::LOG_ERROR(*logger_) << "Creating a TRT engine failed" << std::endl;
        return false;
    }   

    free(buffer);

    return true;
}

bool BaseModel::BuildFromONNX(const std::string& file)
{
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger_));
    if (!builder)
    {
        common::LOG_ERROR(*logger_) << "Creating a builder failed" << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        common::LOG_ERROR(*logger_) << "Creating a network failed" << std::endl;
        return false;
    }

    // TODO: add input arguments (remove)
    //nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    //profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 1080, 1920));
    //profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 1080, 1920));
    //profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 1080, 1920));
    

    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        common::LOG_ERROR(*logger_) << "Creating a config failed" << std::endl;
        return false;
    }
    else {
        config->setMaxWorkspaceSize(4 * (1<<30)); // TODO: configure this
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
        /*config->addOptimizationProfile(profile);*/
    }

	auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *logger_));
	if (!parser)
	{
        common::LOG_ERROR(*logger_) << "Creating a parser failed" << std::endl;
		return false;
	}

    auto parsed = parser->parseFromFile(file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));  // TODO: proper verbosity?
    if (!parsed)
    {
        common::LOG_ERROR(*logger_) << "Parsing an ONNX file failed" << std::endl;
        return false;
    }

    engine_ = builder->buildEngineWithConfig(*network, *config);
    if (!engine_)
    {
        common::LOG_ERROR(*logger_) << "Creating a TRT engine from ONNX failed" << std::endl;
        return false;
    }

    return true;
}

// TODO: alloc - free 구조라면 이 사이에 atomic이 보장되어야 한다
bool BaseModel::AllocDeviceMemory(size_t input_width, size_t input_height, size_t output_width, size_t output_height)
{
    input_width_ = input_width;
    input_height_ = input_height;
    output_width_ = output_width;
    output_height_ = output_height;

    if (!is_built_)
    {
        common::LOG_ERROR(*logger_) << "A TRT engine is not built yet" << std::endl;
        return false;
    }
    if (has_device_memory_)
    {
        common::LOG_ERROR(*logger_) << "Device memory is not freed yet" << std::endl;
        return false;
    }

    size_t input_size = input_width * input_height * 3; // TODO: need to modify if we will use a YUV 420p format
    size_t output_size = output_width * output_height * 3;
    size_t load_size{ 0 }, bytes_per_pixel{ 0 };
    size_t batch_size = model_type_.batch_size;

    if (model_type_.fp16)
        bytes_per_pixel = 2;
    else
        bytes_per_pixel = 4;
    load_size += input_size * batch_size * bytes_per_pixel; // DNN input (fp16/32)
    load_size += output_size * batch_size * bytes_per_pixel; // DNN output (fp16/32)
    load_size += output_size * batch_size * 1; // decoded output in uint8

    bool has_context = (context_ != nullptr);
    switch (model_type_.mtype)
    {
    case MemoryType::kAllocate:
        if (cudaMalloc(&device_buffers_[0], load_size) != cudaSuccess)
        {
            common::LOG_ERROR(*logger_) << "Device malloc failed" << std::endl;
            return false;
        }
        context_ = engine_->createExecutionContext(); // TODO: cudnn관련 error 발생 
        if (!context_)
        {
            FreeDeviceMemory();
            common::LOG_ERROR(*logger_) << "Creating context failed" << std::endl;
            return false;
        }
        device_buffers_[1] = (uint8_t*)device_buffers_[0] + input_size * batch_size * bytes_per_pixel;
        device_buffers_[2] = (uint8_t*)device_buffers_[1] + output_size * batch_size * bytes_per_pixel;
        break;
    case MemoryType::kPreallocate:
        load_size += workspace_size_;
        if ((device_buffers_[0] = device_memory_->Malloc(model_type_.device_id, load_size)) == nullptr)
        {
            common::LOG_ERROR(*logger_) << "Device malloc failed" << std::endl;
            return false;
        }
        if (!has_context)
        {
			context_ = engine_->createExecutionContextWithoutDeviceMemory();
			if (!context_)
			{
				FreeDeviceMemory();
				common::LOG_ERROR(*logger_) << "Creating context failed" << std::endl;
				return false;
			}
        }
        device_buffers_[1] = (uint8_t*)device_buffers_[0] + input_size * batch_size * bytes_per_pixel;
        device_buffers_[2] = (uint8_t*)device_buffers_[1] + output_size * batch_size * bytes_per_pixel;
        //context_->setDeviceMemory((uint8_t*)device_buffers_[0]);
        context_->setDeviceMemory((uint8_t*)device_buffers_[2] + output_size * batch_size);
        break;
    default:
        common::LOG_ERROR(*logger_) << "Invalid memory type" << std::endl;
        return false;
    }

    has_device_memory_ = true;
    load_size_ = load_size;
    return true;
}

void BaseModel::FreeDeviceMemory()
{
    if (!has_device_memory_)
        return;

    switch (model_type_.mtype)
    {
    case MemoryType::kAllocate:
        cudaFree(device_buffers_[0]);
        if (context_)
        {
            context_->destroy(); // TODO: baseline에서 memory free되는지 확인해야함
            context_ = nullptr;
        }
        break;
    case MemoryType::kPreallocate:
        device_memory_->Free(model_type_.device_id, device_buffers_[0]);
        break;
    default:
        common::LOG_ERROR(*logger_) << "Invalid memory type" << std::endl;
        return;
    }

    has_device_memory_ = false;
}

bool BaseModel::Infer(const std::vector<buffers_t>& host_bindings)
{
    if (!has_device_memory_)
    {
        common::LOG_ERROR(*logger_) << "Device memory is not allocated yet" << std::endl;
        return false;
    }

    switch (model_type_.itype)
    {
    case InferType::kSynch:
        if (!InferSynch(host_bindings))
            return false;
        break;
    case InferType::kAsynch:
        if (!InferAsynch(host_bindings))
            return false;
        break;
    default:
        common::LOG_ERROR(*logger_) << "Invalid inference type" << std::endl;
        return false;
    }

    return true;
}
bool BaseModel::InferSynch(const std::vector<buffers_t>& host_bindings)
{
    if (!InferAsynch(host_bindings))
        return false;

    if (cudaStreamSynchronize(stream_) != cudaSuccess)
    {
        common::LOG_ERROR(*logger_) << "cudaStreamSynchronize failed" << std::endl;
        return false;
    }

    return true;
}

bool BaseModel::Synch()
{
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
    {
        common::LOG_ERROR(*logger_) << "cudaStreamSynchronize failed" << std::endl;
        return false;
    }

    return true;
}

// memo (hyunho): maybe, we might add cuda-based yuv-rgb conversion here
bool BaseModel::InferAsynch(const std::vector<buffers_t>& host_bindings)
{
    size_t input_size = input_height_ * input_width_ * 3;
    size_t output_size = output_height_ * output_width_ * 3;
  
    //std::string msg;
    //for (int i = 0; i < context_->getBindingDimensions(0).nbDims; i++)
    //{
    //    msg += std::to_string(context_->getBindingDimensions(0).d[i]) + '\t';
    //}
    //for (int i = 0; i < context_->getBindingDimensions(1).nbDims; i++)
    //{
    //    msg += std::to_string(context_->getBindingDimensions(1).d[i]) + '\t';
    //}
    //std::cout << msg << std::endl;

    for (auto& host_bind : host_bindings)
    {
        if (cudaMemcpyAsync(device_buffers_[2], host_bind.first, input_size * model_type_.batch_size, // TODO: modify it for supporting YUV420p and batch processing
            cudaMemcpyHostToDevice, stream_) != cudaSuccess)
        {
            common::LOG_ERROR(*logger_) << "cudaMemcpyAsynch (host->device) failed" << std::endl;
            return false;
        }

        if (model_type_.fp16)
            uchar2halfArray((uint8_t*)device_buffers_[2], (uint16_t*)device_buffers_[0], input_size, stream_);
        else
            uchar2floatArray((uint8_t*)device_buffers_[2], (uint32_t*)device_buffers_[0], input_size, stream_);

        if (!context_->enqueue(model_type_.batch_size, device_buffers_, stream_, nullptr))
        {
            common::LOG_ERROR(*logger_) << "enqueue failed" << std::endl;
            return false;
        }

        if (model_type_.fp16)
            half2ucharArray((uint16_t*)device_buffers_[1], (uint8_t*)device_buffers_[2], output_size * model_type_.batch_size, stream_);
        else
            float2ucharArray((uint32_t*)device_buffers_[1], (uint8_t*)device_buffers_[2], output_size * model_type_.batch_size, stream_);

        //int result = cudaMemcpyAsync(host_bind.second, device_buffers_[2], output_size * model_type_.batch_size, cudaMemcpyDeviceToHost, stream_);
        //if (result != cudaSuccess)
        if (cudaMemcpyAsync(host_bind.second, device_buffers_[2], output_size * model_type_.batch_size, cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
		{
            common::LOG_ERROR(*logger_) << "cudaMemcpyAsynch (device -> host) failed" << std::endl;
            return false;
		}
	}

    return true;
}

bool BaseModel::InferDone()
{
    if (!has_device_memory_)
        return false;

    if (cudaStreamQuery(stream_) != cudaSuccess)
        return false;

    return true;
}

nvinfer1::IHostMemory* BaseModel::GetSerializedEngine()
{
    return engine_->serialize();
}


size_t BaseModel::GetLoadSize()
{
    return load_size_;
}


