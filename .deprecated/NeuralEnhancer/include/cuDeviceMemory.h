#pragma once

#include <mutex>
#include <map>
#include <vector>
#include <iostream>

class cuDeviceMemory {

private:
	static cuDeviceMemory* pinstance_;
	static std::mutex pmutex_;

protected:
	cuDeviceMemory(const std::vector<int>& device_ids, unsigned long long fragment_size, unsigned long long total_size);
	cuDeviceMemory(const std::vector<int>& device_ids, unsigned long long fragment_size);

	std::map<int, std::vector<void*>> fragments_;
	int num_fragments_;
	unsigned long long fragment_size_;
	std::mutex mutex_;	

public:
	cuDeviceMemory(cuDeviceMemory &other) = delete;
	void operator=(const cuDeviceMemory &) = delete;
	
	static cuDeviceMemory* GetInstance(const std::vector<int>& device_ids, unsigned long long fragment_size, unsigned long long total_size);
	static cuDeviceMemory* GetInstance(const std::vector<int>& device_ids, unsigned long long fragment_size);
	static cuDeviceMemory* GetInstance();

	static void RemoveInstance();
	void* Malloc(int device_id, unsigned long long size);
	void Free(int device_id, void* ptr);
	int get_num_fragments();
};

