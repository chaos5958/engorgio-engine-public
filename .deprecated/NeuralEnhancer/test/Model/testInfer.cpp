#include "BaseModel.h"
#include "testUtility.h"
#include "common.h"
#include "cuDeviceMemory.h"
#include "cuHostMemory.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <string>
#include <thread>

//TODO: measure GPU utilization
//TODO: test witch 2/4/8 batch

namespace fs = std::filesystem;

const std::string DATA_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/data/";
const std::string RESULT_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Model/result/";


double measure_infer_latency(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype, const int num_frames, const int repeat)
{
	BaseModel* bmodel;
	std::chrono::system_clock::time_point end, start;
	std::chrono::duration<double> elapsed_seconds;
	int height = vtype.height, width = vtype.width, scale = dtype.scale;
	buffers_t host_binding;
	std::vector<buffers_t> host_bindings;
	cuHostMemory* hmemory;
	double latency;

	hmemory = cuHostMemory::GetInstance(390000, 10);
	for (int i = 0; i < num_frames; i++)
	{
		host_binding.first = hmemory->Malloc(3 * height * width * mtype.batch_size);
		host_binding.second = hmemory->Malloc(3 * height * scale * width * scale * mtype.batch_size);
		host_bindings.push_back(host_binding);
	}
	bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kERROR);
	if (!bmodel->BuildEngine(dnn_file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
	bmodel->AllocDeviceMemory(width, height, width * scale, height * scale);

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < repeat; i++)
	{
		bmodel->Infer(host_bindings);
	}
	bmodel->Synch();
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	latency = elapsed_seconds.count() / (num_frames * repeat);

	bmodel->FreeDeviceMemory();
	for (int i = 0; i < num_frames; i++)
	{
		hmemory->Free(host_bindings[i].first);
		hmemory->Free(host_bindings[i].second);
	}	
	delete bmodel;

	return latency;
}

void test_infer_latency(const std::string mname, ModelType& mtype, const std::map<VideoType, std::vector<DNNType>>& dtypes_table, const int num_frames, const int repeat, const int chunk_length, const std::vector<int> num_infers)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;
	std::vector<std::string> log_msgs(num_infers.size());
	std::vector<std::ofstream> log_streams(num_infers.size());
	double latency;

	log_file = RESULT_ROOT_DIR + "infer_latency.txt";
	log_stream.open(log_file);
	for (int i = 0; i < num_infers.size(); i++)
	{
		log_file = RESULT_ROOT_DIR + "infer_max_users_i" + std::to_string(num_infers[i]) + ".txt";
		log_streams[i].open(log_file);
	}

	for (auto const& [vtype, dtypes] : dtypes_table)
	{
		for (auto const& dtype : dtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			latency = measure_infer_latency(mtype, dnn_file, vtype, dtype, num_frames, repeat);
			log_msg += std::to_string(latency) + '\t';

			for (int i = 0; i < num_infers.size(); i++)
			{
				log_msgs[i] += std::to_string(chunk_length / (latency * num_infers[i])) + '\t'; 
			}
		}
		log_msg += '\n';
		for (int i = 0; i < num_infers.size(); i++)
		{
			log_msgs[i] += '\n';
		}
	}
	log_stream.write(log_msg.c_str(), log_msg.size());
	log_stream.close();
	for (int i = 0; i < num_infers.size(); i++)
	{
		log_streams[i].write(log_msgs[i].c_str(), log_msgs[i].size());
		log_streams[i].close();
	}
}

void test_0412_meeting()
{
	// Experiment settings
	std::string mname = "EDSR";
	std::map<VideoType, std::vector<DNNType>> dtypes_table;
	dtypes_table[{1280, 720}] = { {8,4,3}, {8,8,3}, {8,16,3}, {8,32,3}, {8,64,3} };
	std::vector<int> num_infers = { 1, 5, 10 };
	int num_frames = 10, repeat = 10, device_id = 3, chunk_length = 4;
	size_t batch_size = 1;
	bool fp16 = true;
	ModelType mtype1 = { BuildType::kTRT, InferType::kAsynch, MemoryType::kPreallocate, fp16, device_id, 1 };

	// Test 1
	cuDeviceMemory* dmemory = cuDeviceMemory::GetInstance({ device_id }, 4 * GB_IN_BYTES);
	test_infer_latency(mname, mtype1, dtypes_table, num_frames, repeat, chunk_length, num_infers);
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