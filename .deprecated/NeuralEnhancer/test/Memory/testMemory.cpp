#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <cuda_runtime.h>
#include "cuHostMemory.h"
#include "cuDeviceMemory.h"

namespace fs = std::filesystem;
const std::string RESULT_ROOT_DIR = "/workspace/research/NeuralEnhancer-data/test/Memory/result/";
const unsigned long long GB = (1 << 30);
const unsigned long long MB = (1 << 20);

enum Target {
	BASELINE,
	OUR
};

void log(std::string log_file, std::vector<double> results)
{
    std::ofstream weightStream(log_file);
    if (!weightStream.is_open())
    {
        std::cerr << "File open failed" << std::endl;
        return;
    }

    for (auto& result : results)
    {
		std::string log = std::to_string(result);
		weightStream.write(log.c_str(), log.size());
		weightStream.write("\t", 1);
    }
}

std::string get_target_name(Target target)
{
    std::string name;

    switch (target)
    {
    case Target::BASELINE:
        name = "baseline";
        break;
    case Target::OUR:
        name = "our";
        break;
    }

    return name;
}


void test_device_memory_target(Target target, int device_id, int num_tests, std::vector<unsigned long long> fragment_sizes) 
{
    void* ptr;
    const std::vector<int> device_ids{ device_id };

    std::chrono::system_clock::time_point end, start;
    std::chrono::duration<double> elapsed_seconds;
    std::vector<double> latencies;
    cuDeviceMemory* dmemory = nullptr;

    for (auto& fragment_size : fragment_sizes)
    {
        if (target == Target::OUR)
            dmemory = cuDeviceMemory::GetInstance(device_ids, fragment_size);
        
        start = std::chrono::high_resolution_clock::now();
		switch (target)
		{
		case Target::BASELINE:
			for (int i = 0; i < num_tests; i++)
			{
				if (cudaMalloc(&ptr, fragment_size) != cudaSuccess)
				{
					std::cerr << "cudaMalloc failed" << std::endl;
					return;

				}
				cudaFree(ptr);
			}
			break;
		case Target::OUR:
			for (int i = 0; i < num_tests; i++)
			{
				ptr = dmemory->Malloc(device_id, fragment_size);
				dmemory->Free(device_id, ptr);
			}
			break;
		}
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        latencies.push_back(elapsed_seconds.count() / num_tests);

        if (target == Target::OUR)
            cuDeviceMemory::RemoveInstance();
    }

    std::string log_file = RESULT_ROOT_DIR + get_target_name(target) + "_device_memory.txt";
    log(log_file, latencies);
}

void test_host_memory_target(Target target, int num_tests, std::vector<unsigned long long> fragment_sizes)
{
    void* ptr;

    std::chrono::system_clock::time_point end, start;
    std::chrono::duration<double> elapsed_seconds;
    std::vector<double> latencies;
    cuHostMemory* hmemory = nullptr;

    for (auto& fragment_size : fragment_sizes)
    {
        if (target == Target::OUR)
            hmemory = cuHostMemory::GetInstance(390000, 10);
        start = std::chrono::high_resolution_clock::now();

		switch (target)
		{
		case Target::BASELINE:
			for (int i = 0; i < num_tests; i++)
			{
                if (cudaHostAlloc(&ptr, fragment_size, cudaHostAllocDefault) != cudaSuccess)
                {
					std::cerr << "cudaMalloc failed" << std::endl;
					return;

				}
				cudaFreeHost(ptr);
			}
			break;
		case Target::OUR:
			for (int i = 0; i < num_tests; i++)
			{
				ptr = hmemory->Malloc(fragment_size);
				hmemory->Free(ptr);
			}
			break;
		}
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        latencies.push_back(elapsed_seconds.count() / num_tests);

        if (target == Target::OUR)
            cuHostMemory::RemoveInstance();
    }

    std::string log_file = RESULT_ROOT_DIR + get_target_name(target) + "_host_memory.txt";
    log(log_file, latencies);
}

//04.12 미팅 준비 용 테스트 코드
void test_0412_meeting()
{
    // Experiment settings
    int num_tests, device_id;
    std::vector<unsigned long long> device_fragment_sizes{ 1 * GB, 2 * GB, 4 * GB, 8 * GB };
    std::vector<unsigned long long> host_fragment_sizes{ 1280 * 720 * 3, 1920 * 1080 * 3, 2560 * 1440 * 3, 3840 * 2160 * 3 };

    // Create directories
    if (!fs::exists(RESULT_ROOT_DIR))
        fs::create_directories(RESULT_ROOT_DIR);

    // Test 1: Host memory
    num_tests = 50;
    test_host_memory_target(Target::BASELINE, 5, host_fragment_sizes); // dummy
    test_host_memory_target(Target::BASELINE, num_tests, host_fragment_sizes);
    test_host_memory_target(Target::OUR, num_tests, host_fragment_sizes);

    // Test 2: Device memory
    num_tests = 50;
    device_id = 0;
    test_device_memory_target(Target::BASELINE, device_id, 5, device_fragment_sizes); // dummy
    test_device_memory_target(Target::BASELINE, device_id, num_tests, device_fragment_sizes);
    test_device_memory_target(Target::OUR, device_id, num_tests, device_fragment_sizes);
}

int main()
{
    test_0412_meeting();
    return 0;
}