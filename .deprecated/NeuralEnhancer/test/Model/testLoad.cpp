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

namespace fs = std::filesystem;

const std::string DATA_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/data/";
const std::string RESULT_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/result/";

size_t measure_load_memory(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype)
{
	BaseModel* bmodel;
	size_t memory;
	int height = vtype.height, width = vtype.width, scale = dtype.scale;

	bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kERROR);
	if (!bmodel->BuildEngine(dnn_file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
	bmodel->AllocDeviceMemory(height, width, height * scale, width * scale);
	bmodel->FreeDeviceMemory();
	memory = bmodel->GetLoadSize();
	delete bmodel;

	return memory;
}

void test_load_memory(const std::string mname, ModelType& mtype, const std::map<VideoType, std::vector<DNNType>>& dtypes_table)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;

	size_t memory;
	log_file = RESULT_ROOT_DIR + "load_memory.txt";
	log_stream.open(log_file);
	for (auto const& [vtype, dtypes] : dtypes_table)
	{
		for (auto const& dtype : dtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			memory = measure_load_memory(mtype, dnn_file, vtype, dtype);
			log_msg += std::to_string(memory / MB_IN_BYTES) + '\t';
		}
		log_msg += '\n';
	}
	log_stream.write(log_msg.c_str(), log_msg.size());
	log_stream.close();
}

double measure_load_latency(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype, const int num_tests)
{
	BaseModel* bmodel;
	std::chrono::system_clock::time_point end, start;
	std::chrono::duration<double> elapsed_seconds;
	int height, width, scale;
	double latency;

	bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kERROR);
	if (!bmodel->BuildEngine(dnn_file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
	height = vtype.height;
	width = vtype.width;
	scale = dtype.scale;

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_tests; i++)
	{
		bmodel->AllocDeviceMemory(height, width, height * scale, width * scale);
		bmodel->FreeDeviceMemory();

	}
	delete bmodel;
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	latency = elapsed_seconds.count() / num_tests;

	return latency;
}

void test_load_latency(const std::string mname, ModelType& mtype, const std::map<VideoType, std::vector<DNNType>>& dtypes_table, const int num_tests)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;
	
	double latency;
	log_file = RESULT_ROOT_DIR + "load_latency_" + get_mtype_name(mtype.mtype) + ".txt";
	log_stream.open(log_file);
	for (auto const& [vtype, dtypes] : dtypes_table)
	{
		for (auto const& dtype : dtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			latency = measure_load_latency(mtype, dnn_file, vtype, dtype, num_tests);
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
	int num_tests = 50, device_id = 0;
	size_t batch_size = 1;
	bool fp16 = true;
	ModelType mtype1 = {BuildType::kTRT, InferType::kAsynch, MemoryType::kAllocate, fp16, device_id, batch_size };
	ModelType mtype2 = { BuildType::kTRT, InferType::kAsynch, MemoryType::kPreallocate, fp16, device_id, batch_size };
	
	// Test 1
	test_load_latency(mname, mtype1, dtypes_table, num_tests);
	cuDeviceMemory* dmemory = cuDeviceMemory::GetInstance({ device_id }, 4 * GB_IN_BYTES);
	test_load_latency(mname, mtype2, dtypes_table, num_tests);
	cuDeviceMemory::RemoveInstance();

	// Test 2
	dmemory = cuDeviceMemory::GetInstance({ device_id }, 4 * GB_IN_BYTES);
	test_load_memory(mname, mtype2, dtypes_table);
	cuDeviceMemory::RemoveInstance();
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