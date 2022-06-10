#pragma once

#include <iostream>
#include <mutex>
#include <vector>
#include <map>
#include <deque>

size_t get_frame_size(int resolution);

struct cuFramePool
{
	std::deque<void*> frames;
	std::mutex mutex;
	int num_frames;
};

class cuHostMemoryV2 {

private:
	static cuHostMemoryV2* pinstance_;
	static std::mutex pmutex_;

public:
	std::map<int, cuFramePool*> frame_pools_;
	
	cuHostMemoryV2(int num_gpus);
	cuHostMemoryV2(cuHostMemoryV2& other) = delete;
	void operator=(const cuHostMemoryV2&) = delete;

	static cuHostMemoryV2* GetInstance(int num_gpus); 
	static cuHostMemoryV2* GetInstance(); 
	static void RemoveInstance();
	void* Malloc(int resolution);
	void Free(int resolution, void* ptr);
	int NumFrames(int resolution);
};

