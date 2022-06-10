#include "BaseModel.h"
#include "cuDeviceMemory.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fstream>


const std::string PROJECT_ROOT = "../../../../../../";
const std::string DATA_ROOT_DIR = PROJECT_ROOT + "src/NeuralEnhancer/test/Model/data/";

void test_onnx(ModelType &mtype, const std::string &onnx_file, const std::string &trt_file)
{
	BaseModel* bmodel = new BaseModel(mtype);
	std::chrono::system_clock::time_point end, start;
	std::chrono::duration<double> elapsed_seconds;

	start = std::chrono::high_resolution_clock::now();
	if (!bmodel->BuildEngine(trt_file))

	if (!bmodel->BuildEngine(onnx_file))
	{
		std::cerr << "Build a TRT engine from ONNX failed" << std::endl;
		return;
	}

	std::cout << trt_file << std::endl;

	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	std::cout << "Elapsed time: " << elapsed_seconds.count() * 1000 << "ms" << std::endl;

	nvinfer1::IHostMemory* hostEngine = bmodel->GetSerializedEngine();
	std::cout << "size: " << hostEngine->size() << std::endl;
		
	std::ofstream weightStream(trt_file, std::ifstream::binary);

	if (!weightStream.is_open())
		return;

	weightStream.write((char*)hostEngine->data(), hostEngine->size());
	hostEngine->destroy();

	std::cout << "ONNX: pass all" << std::endl;
}

void test_trt(ModelType& mtype, const std::string& trt_file)
{
	BaseModel* bmodel = new BaseModel(mtype);

	std::chrono::system_clock::time_point end, start;
	std::chrono::duration<double> elapsed_seconds;

	start = std::chrono::high_resolution_clock::now();
	if (!bmodel->BuildEngine(trt_file))
	{
		std::cerr << "Build a TRT engine failed" << std::endl;
		return;
	}
	std::cout << "TRT: pass all" << std::endl;
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	std::cout << "Elapsed time: " << elapsed_seconds.count() * 1000 << "ms" << std::endl;
}

int main()
{
	ModelType trt_mtype = {
		BuildType::kTRT,
		InferType::kSynch,
		MemoryType::kPreallocate,
		true,
		0,
		1
	};

	ModelType onnx_mtype = {
		BuildType::kONNX,
		InferType::kSynch,
		MemoryType::kPreallocate,
		true,
		0,
		1
	};
	const std::string onnx_file = DATA_ROOT_DIR + "720p/EDSR_B8_F4_S3/EDSR_B8_F4_S3.onnx";
	const std::string trt_file = "/workspace/research/EDSR_B8_F4_S3.plan";

	ModelType pytorch_mtype = {
		BuildType::kPyTorch,
		InferType::kSynch,
		MemoryType::kPreallocate,
		true,
		0,
		1
	};

	std::vector<int> device_ids{ 0 };
	cuDeviceMemory* dmemory = cuDeviceMemory::GetInstance(device_ids, (size_t)1 << 31);
	test_onnx(onnx_mtype, onnx_file, trt_file);
	test_trt(trt_mtype, trt_file);
}