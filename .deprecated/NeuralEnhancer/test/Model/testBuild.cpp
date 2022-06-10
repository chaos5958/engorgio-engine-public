#include "BaseModel.h"
#include "testUtility.h"
#include "common.h"
#include "cuDeviceMemory.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <string>
#include <thread>
#include <nvml.h>

using namespace std::string_literals;
namespace fs = std::filesystem;

const std::string DATA_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/data/";
const std::string RESULT_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/result/";

std::pair<size_t, int> measure_build_memory(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype)
{
	BaseModel* bmodel;
	size_t model_memory;
	int max_users;
	int height = vtype.height, width = vtype.width, scale = dtype.scale;
	cuDeviceMemory *dmemory;
	nvmlDevice_t device;
	nvmlMemory_t gpu_memory;
	nvmlReturn_t result;
		
	result = nvmlInit();
	if (NVML_SUCCESS != result)
	{
		throw std::runtime_error("Failed to initialize NVML: "s + nvmlErrorString(result) + "\n"s);
	}

	// Get model size
	if (mtype.mtype == MemoryType::kPreallocate)
		dmemory = cuDeviceMemory::GetInstance({ mtype.device_id }, 4 * GB_IN_BYTES);
	bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kERROR);
	if (!bmodel->BuildEngine(dnn_file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
	bmodel->AllocDeviceMemory(height, width, height * scale, width * scale);
	bmodel->FreeDeviceMemory();
	model_memory = bmodel->GetLoadSize();
	delete bmodel;
	if (mtype.mtype == MemoryType::kPreallocate)
		cuDeviceMemory::RemoveInstance();

	// Calculate max users
	result = nvmlDeviceGetHandleByIndex(mtype.device_id, &device);
	if (NVML_SUCCESS != result)
	{
		throw std::runtime_error("Failed to get handle for device "s + std::to_string(mtype.device_id) + ": " + nvmlErrorString(result) + "\n"s);
	}
	result = nvmlDeviceGetMemoryInfo(device, &gpu_memory);
	if (NVML_SUCCESS != result)
	{
		throw std::runtime_error("Failed to get information of device "s + std::to_string(mtype.device_id) + ": " + nvmlErrorString(result) + "\n"s);
	}
	max_users = (gpu_memory.total / model_memory);

	return std::pair<size_t, int>(model_memory, max_users);
}

void test_build_memory(const std::string mname, ModelType& mtype, const std::map<VideoType, std::vector<DNNType>>& dtypes_table)
{
	std::string log_msg1, log_msg2, log_file, dnn_file;
	std::ofstream log_stream1, log_stream2;
	std::pair<size_t, int> result;

	log_file = RESULT_ROOT_DIR + "build_memory.txt";
	log_stream1.open(log_file);
	log_file = RESULT_ROOT_DIR + "build_max_users.txt";
	log_stream2.open(log_file);
	for (auto const& [vtype, dtypes] : dtypes_table)
	{
		for (auto const& dtype : dtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			result = measure_build_memory(mtype, dnn_file, vtype, dtype);
			log_msg1 += std::to_string(result.first / MB_IN_BYTES) + '\t';
			log_msg2 += std::to_string(result.second) + '\t';
		}
		log_msg1 += '\n';
		log_msg2 += '\n';
	}
	log_stream1.write(log_msg1.c_str(), log_msg1.size());
	log_stream2.write(log_msg2.c_str(), log_msg2.size());
	log_stream1.close();
	log_stream2.close();
}

double measure_build_latency(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype, const int num_tests)
{
	BaseModel* bmodel;
	std::chrono::system_clock::time_point end, start;
	std::chrono::duration<double> elapsed_seconds;
	int height = vtype.height, width = vtype.width, scale = dtype.scale;
	double latency;
	cuDeviceMemory* dmemory;

	if (mtype.mtype == MemoryType::kPreallocate)
		dmemory = cuDeviceMemory::GetInstance({ mtype.device_id }, 4 * GB_IN_BYTES);

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_tests; i++)
	{
		bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kERROR);
		if (!bmodel->BuildEngine(dnn_file))
			throw std::runtime_error("Build a TRT engine from ONNX failed");
		delete bmodel;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	latency = elapsed_seconds.count() / num_tests;

	if (mtype.mtype == MemoryType::kPreallocate)
		cuDeviceMemory::RemoveInstance();

	return latency;
}

void test_build_latency(const std::string mname, ModelType& mtype, const std::map<VideoType, std::vector<DNNType>>& dtypes_table, const int num_tests)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;

	double latency;
	log_file = RESULT_ROOT_DIR + "build_latency_" + get_btype_name(mtype.btype) + ".txt";
	log_stream.open(log_file);
	for (auto const& [vtype, dtypes] : dtypes_table)
	{
		for (auto const& dtype : dtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			latency = measure_build_latency(mtype, dnn_file, vtype, dtype, num_tests);
			log_msg += std::to_string(latency) + '\t';
		}
		log_msg += '\n';
	}
	log_stream.write(log_msg.c_str(), log_msg.size());
	log_stream.close();
}

void test_0412_meeting()
{
	// Experiment settings
	std::string mname = "EDSR";
	std::map<VideoType, std::vector<DNNType>> dtypes_table;
	dtypes_table[{1280, 720}] = { {8,4,3}, {8,8,3}, {8,16,3}, {8,32,3}, {8,64,3} };
	int num_tests = 10, device_id = 3;
	size_t batch_size = 1;
	bool fp16 = true;
	ModelType mtype1 = { BuildType::kONNX, InferType::kAsynch, MemoryType::kAllocate, fp16, device_id, batch_size };
	ModelType mtype2 = { BuildType::kTRT, InferType::kAsynch, MemoryType::kPreallocate, fp16, device_id, batch_size };

	// Test 1
	test_build_memory(mname, mtype2, dtypes_table);

	// Test 2
	test_build_latency(mname, mtype1, dtypes_table, num_tests);
	test_build_latency(mname, mtype2, dtypes_table, num_tests);
}

int main()
{
	// Create directories
	if (!fs::exists(RESULT_ROOT_DIR))
		fs::create_directories(RESULT_ROOT_DIR);

	// Run tests
	test_0412_meeting();

	return 0;
}