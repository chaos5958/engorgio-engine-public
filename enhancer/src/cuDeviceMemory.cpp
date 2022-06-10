#include "cuDeviceMemory.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <string>
using namespace std::string_literals;

cuDeviceMemory* cuDeviceMemory::pinstance_{ nullptr };
std::mutex cuDeviceMemory::pmutex_;

cuDeviceMemory::cuDeviceMemory(const std::vector<int>& device_ids, unsigned long long fragment_size, unsigned long long total_size)
{
    void* ptr;

    num_fragments_ = total_size / fragment_size;
    fragment_size_ = fragment_size;

    for (auto id : device_ids)
    {
        if (cudaSetDevice(id) != cudaSuccess)
            throw std::runtime_error("GPU is not available");
        
        fragments_.insert(std::pair<int, std::vector<void*>>(id, std::vector<void*>()));
        for (int i = 0; i < num_fragments_; i++)
        {
            if (cudaMalloc(&ptr, fragment_size) != cudaSuccess)
                throw std::runtime_error("GPU memory allocation fails");
            fragments_[id].push_back(ptr);
        }
    }
}

cuDeviceMemory::cuDeviceMemory(const std::vector<int>& device_ids, unsigned long long fragment_size)
{
    void* ptr;
    nvmlDevice_t device;
    nvmlMemory_t memory;

    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        throw std::runtime_error("Failed to initialize NVML: "s + nvmlErrorString(result) + "\n"s);
    }

    for (auto id : device_ids)
    {
        result = nvmlDeviceGetHandleByIndex(id, &device);
        if (NVML_SUCCESS != result)
        {
            throw std::runtime_error("Failed to get handle for device "s + std::to_string(id) + ": " + nvmlErrorString(result) + "\n"s);
        }
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS != result)
        {
            throw std::runtime_error("Failed to get information of device "s + std::to_string(id) + ": " + nvmlErrorString(result) + "\n"s);
        }

        num_fragments_ = (memory.free * DEVICE_MEM_FRAC / fragment_size); 
        fragment_size_ = fragment_size;

        if (cudaSetDevice(id) != cudaSuccess)
            throw std::runtime_error("GPU is not available");

        fragments_.insert(std::pair<int, std::vector<void*>>(id, std::vector<void*>()));
        for (int i = 0; i < num_fragments_; i++)
        {
            if (cudaMalloc(&ptr, fragment_size) != cudaSuccess)
                throw std::runtime_error("GPU memory allocation fails");
            fragments_[id].push_back(ptr);
        }
    }
}

cuDeviceMemory::cuDeviceMemory(const std::vector<int>& device_ids)
{
    void* ptr;
    nvmlDevice_t device;
    nvmlMemory_t memory;
    num_fragments_ = 2;

    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        throw std::runtime_error("Failed to initialize NVML: "s + nvmlErrorString(result) + "\n"s);
    }

    for (auto id : device_ids)
    {
        result = nvmlDeviceGetHandleByIndex(id, &device);
        if (NVML_SUCCESS != result)
        {
            throw std::runtime_error("Failed to get handle for device "s + std::to_string(id) + ": " + nvmlErrorString(result) + "\n"s);
        }
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS != result)
        {
            throw std::runtime_error("Failed to get information of device "s + std::to_string(id) + ": " + nvmlErrorString(result) + "\n"s);
        }

        fragment_size_ = (memory.free * DEVICE_MEM_FRAC) / num_fragments_;

        if (cudaSetDevice(id) != cudaSuccess)
            throw std::runtime_error("GPU is not available");

        fragments_.insert(std::pair<int, std::vector<void*>>(id, std::vector<void*>()));
        for (int i = 0; i < num_fragments_; i++)
        {
            if (cudaMalloc(&ptr, fragment_size_) != cudaSuccess)
                throw std::runtime_error("GPU memory allocation fails");
            fragments_[id].push_back(ptr);
        }
    }
}

cuDeviceMemory* cuDeviceMemory::GetInstance(const std::vector<int>& device_ids, unsigned long long fragment_size, unsigned long long total_size)
{
	std::lock_guard<std::mutex> lock(pmutex_);
	if (pinstance_ == nullptr)
	{
		pinstance_ = new cuDeviceMemory(device_ids, fragment_size, total_size);
	}
	return pinstance_;
}

cuDeviceMemory* cuDeviceMemory::GetInstance(const std::vector<int>& device_ids, unsigned long long fragment_size)
{
    std::lock_guard<std::mutex> lock(pmutex_);
    if (pinstance_ == nullptr)
    {
        pinstance_ = new cuDeviceMemory(device_ids, fragment_size);
    }
    return pinstance_;
}


cuDeviceMemory* cuDeviceMemory::GetInstance(const std::vector<int>& device_ids)
{
    std::lock_guard<std::mutex> lock(pmutex_);
    if (pinstance_ == nullptr)
    {
        pinstance_ = new cuDeviceMemory(device_ids);
    }
    return pinstance_;
}

cuDeviceMemory* cuDeviceMemory::GetInstance()
{
    std::lock_guard<std::mutex> lock(pmutex_);
    return pinstance_;
}

void cuDeviceMemory::RemoveInstance()
{
    std::lock_guard<std::mutex> lock(pmutex_);
    std::map<int, std::vector<void*>>::iterator it;
    if (pinstance_ != nullptr)
    {
        // free CUDA memory
        for (it = pinstance_->fragments_.begin(); it != pinstance_->fragments_.end(); it++)
        {
            if (cudaSetDevice(it->first) != cudaSuccess)
                throw std::runtime_error("GPU is not available");

            for (auto& fragment : it->second)
            {
                cudaFree(fragment);
            }
        }

        // free the singletone instance
        delete pinstance_;
        pinstance_ = nullptr;
    }
}

void* cuDeviceMemory::Malloc(int device_id, unsigned long long size)
{
    std::lock_guard<std::mutex> lock_guard(mutex_);
    if (size > fragment_size_ || fragments_.find(device_id) == fragments_.end() || fragments_[device_id].empty())
    {
        return nullptr;
    }

    void* ptr = fragments_[device_id].back();
    fragments_[device_id].pop_back();
    return ptr;
}

void cuDeviceMemory::Free(int device_id, void* ptr)
{
    std::lock_guard<std::mutex> lock_guard(mutex_);
    if (fragments_.find(device_id) != fragments_.end())
        fragments_[device_id].push_back(ptr);
}

int cuDeviceMemory::get_num_fragments() {
	return num_fragments_;
}
