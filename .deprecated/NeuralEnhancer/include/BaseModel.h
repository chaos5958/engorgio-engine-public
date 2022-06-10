#pragma once

#include <vector>
#include "NvInfer.h"
#include "logging.h"
#include "common.h"
#include "cuDeviceMemory.h"
#include <string>
#include <NvOnnxParser.h>
#include <memory>

enum class BuildType : int
{
	kTRT = 0,
	kONNX = 1,
	kPyTorch = 2
};

enum class InferType : int
{
	kSynch = 0,
	kAsynch = 1
};

enum class MemoryType : int
{
	kAllocate = 0,
	kPreallocate = 1
};


struct ModelType
{
	BuildType btype;
	InferType itype;
	MemoryType mtype;
	bool fp16;
	int device_id;
	size_t batch_size;
};

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

using buffers_t = std::pair <void*, void*>; // input, output buffer pair

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

//TODO: Apply smartpointer, and Check whether it works properly
class BaseModel
{
private:
	static common::Logger shared_logger_;

protected: 
	ModelType model_type_;
	bool has_device_memory_{ false };
	bool is_built_{ false };
	size_t workspace_size_, load_size_;
	size_t input_width_, input_height_, output_width_, output_height_;
	void* device_buffers_[3]{ nullptr, nullptr, nullptr };
	cuDeviceMemory* device_memory_{ nullptr };

	// TensorRT, CUDA variables
	common::Logger* logger_{ nullptr };
	nvinfer1::ICudaEngine* engine_{ nullptr };
	nvinfer1::IExecutionContext* context_{ nullptr };
	cudaStream_t stream_{ nullptr };

	// Fuctions for building TensorRT engines
	bool BuildFromTRT(const std::string& file);
	bool BuildFromONNX(const std::string& file);
	//virtual bool BuildFromPyTorch(const std::string& file);
	size_t ReadFile(const std::string& file, void** buffer);

	bool InferSynch(const std::vector<buffers_t>& host_bindings);
	bool InferAsynch(const std::vector<buffers_t>& host_bindings);
	
public:
	BaseModel(ModelType& model_type, nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO);
	~BaseModel();

	// Functions for building TensorRT engines
	bool BuildEngine(const std::string& file);
	void DestroyEngine();

	// Functions for allocating/freeing CUDA memory
	bool AllocDeviceMemory(size_t input_width, size_t input_height, size_t output_width, size_t output_height);
	void FreeDeviceMemory();

	bool Infer(const std::vector<buffers_t>& host_bindings);
	bool InferDone();
	bool Synch();

	nvinfer1::IHostMemory* GetSerializedEngine();
	size_t GetLoadSize();
};