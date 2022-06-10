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

void build_mt(BaseModel* bmodel, std::string& file)
{
	if (!bmodel->BuildEngine(file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
}

struct Result
{
	double latency;
	double throughput;
	std::vector<std::pair<double, int>> measurements;
};

Result measure_throughput(ModelType& mtype, const std::string dnn_file, const VideoType& vtype, const DNNType& dtype,const int num_infer, const int duration)
{
	BaseModel* bmodel;
	std::chrono::system_clock::time_point end, start, start_;
	std::chrono::duration<double> elapsed_seconds;
	int height = vtype.height, width = vtype.width, scale = dtype.scale;
	buffers_t host_binding;
	std::vector<buffers_t> host_bindings;
	cuHostMemory* hmemory;
	cuDeviceMemory* dmemory;
	Result result;

	// 1. Allocate host memory
	hmemory = cuHostMemory::GetInstance(390000, 10);
	for (int i = 0; i < num_infer; i++)
	{
		host_binding.first = hmemory->Malloc(3 * height * width * mtype.batch_size);
		host_binding.second = hmemory->Malloc(3 * height * scale * width * scale * mtype.batch_size);
		host_bindings.push_back(host_binding);
	}
	dmemory = cuDeviceMemory::GetInstance({ mtype.device_id }, 4 * GB_IN_BYTES);

	// 2. Build models
	bmodel = new BaseModel(mtype, nvinfer1::ILogger::Severity::kINFO);

	// TODO: Use a thread pool to enablem multi-threading
	// 3. Build engines
	start = std::chrono::high_resolution_clock::now();
	if (!bmodel->BuildEngine(dnn_file))
		throw std::runtime_error("Build a TRT engine from ONNX failed");
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	result.latency = elapsed_seconds.count();

	// TODO: Asynchronous processing for infer and synch
	start_ = std::chrono::high_resolution_clock::now();
	// 4. Load, Infer
	int log_interval = 10, count = 1, total_frames = 0;
	while (1)
	{
		bmodel->AllocDeviceMemory(width, height, width * scale, height * scale);
		bmodel->Infer(host_bindings);
		bmodel->FreeDeviceMemory();

		if (count % log_interval == 0)
		{
			end = std::chrono::high_resolution_clock::now();
			elapsed_seconds = end - start;
			total_frames += num_infer * log_interval;
			result.measurements.push_back(std::pair<double, int>(elapsed_seconds.count(), total_frames));

			elapsed_seconds = end - start_;
			if (elapsed_seconds.count() > duration)
				break;
		}

		count += 1;
	}

	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start_;
	result.throughput = (num_infer * count) / elapsed_seconds.count();

	// 5. Free host & device memory
	for (int i = 0; i < num_infer; i++)
	{
		hmemory->Free(host_bindings[i].first);
		hmemory->Free(host_bindings[i].second);
	}
	cuDeviceMemory::RemoveInstance();

	// 6. Destroy models
	delete bmodel;

	return result;
}

std::pair<int, std::vector<double>>  estimate_users_and_throughput(const std::string mname, ModelType& mtype, const VideoType& vtype, const DNNType& dtype, const int num_infer, const int duration, const int chunk_length, const int stream_length)
{
	std::vector<double> throughputs;
	std::string dnn_file;
	int num_users;
	double throughput, min_throughput;
	Result result;

	dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
	result = measure_throughput(mtype, dnn_file, vtype, dtype, num_infer, duration);

	num_users = 1;
	while (1)
	{
		min_throughput = (num_infer * num_users) / chunk_length;
		throughput = ((stream_length - result.latency * num_users) * result.throughput) / stream_length;
		if (throughput < min_throughput)
			break;

		num_users += 1;
	}

	for (num_users = 1; num_users < 50; num_users++)
	{
		throughput = ((stream_length - result.latency * num_users) * result.throughput) / stream_length;
		if (throughput < 0)
			throughput = 0;
		throughputs.push_back(throughput);
	}

	std::cout << "Latency: " << std::to_string(result.latency) << std::endl;
	std::cout << "Throuhgput: " << std::to_string(result.throughput) << std::endl;

	return std::pair<int, std::vector<double>>(num_users, throughputs);
}

int estimate_max_users(const Result& result, const int num_infer, const int duration, const int chunk_length, const int stream_length)
{
	int num_users = 1;
	double throughput, min_throughput;

	while (1)
	{
		min_throughput = (num_infer * num_users) / chunk_length;
		throughput = ((stream_length - result.latency * num_users) * result.throughput) / stream_length;
		if (throughput < min_throughput)
			break;

		num_users += 1;
	}

	return num_users;
}

int test_max_users(const std::vector<Result>& results, const int num_infer, const int duration, const int chunk_length, const int stream_length)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;
	int max_users;

	log_file = RESULT_ROOT_DIR + "stream_max_users_i" + std::to_string(num_infer) + ".txt";
	log_stream.open(log_file);

	for (auto& result : results)
	{
		max_users = estimate_max_users(result, num_infer, duration, chunk_length, stream_length);
		log_msg += std::to_string(max_users) + '\t';
	}

	log_stream.write(log_msg.c_str(), log_msg.size());
	log_stream.close();
}

std::vector<double> estimate_throuhgput(const Result& result, const int num_infer, const int duration, const int chunk_length, const int stream_length)
{
	std::vector<double> throughputs;
	double throughput;
	int num_users;

	for (num_users = 1; num_users < 50; num_users++)
	{
		throughput = ((stream_length - result.latency * num_users) * result.throughput) / stream_length;
		if (throughput < 0)
			throughput = 0;
		throughputs.push_back(throughput);
	}

	return throughputs;
}

int test_throughput(const std::vector<Result>& results, const int num_infer, const int duration, const int chunk_length, const int stream_length)
{
	std::string log_msg, log_file, dnn_file;
	std::ofstream log_stream;
	std::vector<std::vector<double>> throughputs;

	log_file = RESULT_ROOT_DIR + "stream_throughput_i" + std::to_string(num_infer) + ".txt";
	log_stream.open(log_file);

	for (auto& result : results)
	{
		throughputs.push_back(estimate_throuhgput(result, num_infer, duration, chunk_length, stream_length));
	}

	for (int i = 0; i < throughputs[0].size(); i++)
	{
		log_msg += std::to_string(i + 1) + '\t';
		for (auto& throughput : throughputs)
		{
			log_msg += std::to_string(throughput[i]) + '\t';
		}
		log_msg += '\n';
	}

	log_stream.write(log_msg.c_str(), log_msg.size());
	log_stream.close();
}

void test_0503_meeting()
{
	// Experiment settings
	std::string mname = "EDSR", dnn_file;
	int device_id = 3;
	bool fp16 = true;
	std::vector<ModelType> mtypes;
	mtypes.push_back(ModelType{ BuildType::kONNX, InferType::kSynch, MemoryType::kAllocate, fp16, device_id, 1 });
	mtypes.push_back(ModelType{ BuildType::kTRT, InferType::kSynch, MemoryType::kAllocate, fp16, device_id, 1 });
	mtypes.push_back(ModelType{ BuildType::kONNX, InferType::kSynch, MemoryType::kPreallocate, fp16, device_id, 1 });
	mtypes.push_back(ModelType{ BuildType::kTRT, InferType::kSynch, MemoryType::kPreallocate, fp16, device_id, 1 });
	VideoType vtype{ 1280, 720 };
	DNNType dtype{ 8, 16, 3 };
	std::vector<int> num_infers{ 1, 5, 10 };
	int duration = 5, chunk_length = 4, stream_length = 300;
	std::vector<Result> results;

	for (auto& num_infer : num_infers)
	{
		for (auto& mtype : mtypes)
		{
			dnn_file = get_dnn_file(DATA_ROOT_DIR, vtype, get_dnn_name(mname, mtype.btype, dtype));
			results.push_back(measure_throughput(mtype, dnn_file, vtype, dtype, num_infer, duration));
		}
		test_max_users(results, num_infer, duration, chunk_length, stream_length);
		test_throughput(results, num_infer, duration, chunk_length, stream_length);
	}
	
}

int main()
{
	// Create directories
	if (!fs::exists(RESULT_ROOT_DIR))
		fs::create_directories(RESULT_ROOT_DIR);

	// Run tests
	test_0503_meeting();

	return 0;
}
