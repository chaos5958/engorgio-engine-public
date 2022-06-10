#pragma once

#include <iostream>
#include <mutex>
#include <vector>
#include <map>

struct FragmentHeader
{
	FragmentHeader* prev;
	FragmentHeader* next;
	uint64_t divided; // used to determine whether current fragment can be merged with its neightbors
	int log_scale;

	FragmentHeader* prev_free;
	FragmentHeader* next_free;
	bool allocated;
};

inline void InsertFragment(std::vector<FragmentHeader*>& fragments, FragmentHeader* fragment)
{
	FragmentHeader* prev_fragment;
	int log_scale = fragment->log_scale;

	prev_fragment = fragments[log_scale];
	if (prev_fragment != nullptr)
		prev_fragment->next_free = fragment;
	fragment->next_free = nullptr;
	fragment->prev_free = prev_fragment;
	fragment->allocated = false;
	fragments[log_scale] = fragment;
}

inline FragmentHeader* PopFragment(std::vector<FragmentHeader*>& fragments, int log_scale)
{
	FragmentHeader* curr_last_fragment, * next_last_fragment;

	curr_last_fragment = fragments[log_scale];
	next_last_fragment = curr_last_fragment->prev_free;
	if (next_last_fragment != nullptr)
		next_last_fragment->next_free = nullptr;
	fragments[log_scale] = next_last_fragment;
	curr_last_fragment->allocated = true;

	return curr_last_fragment;
}

inline void RemoveFragment(std::vector<FragmentHeader*>& fragments, FragmentHeader* fragment)
{
	FragmentHeader* prev_fragment = fragment->prev_free;
	FragmentHeader* next_fragment = fragment->next_free;

	if (next_fragment != nullptr)
		next_fragment->prev_free = prev_fragment;
	else
		fragments[fragment->log_scale] = prev_fragment;
	if (prev_fragment != nullptr)
		prev_fragment->next_free = next_fragment;

	fragment->allocated = true;
}

class cuHostMemory {

private:
	static cuHostMemory* pinstance_;
	static std::mutex pmutex_;

protected:
	const int min_size_;
	const int max_log_scale_;
	size_t total_size_;
	std::mutex mutex_;
	std::vector<FragmentHeader*> fragments_;
	std::vector<void*> ptrs_;

	cuHostMemory(size_t min_size, int max_log_scale);
	bool cudaMalloc();
public:
	cuHostMemory(cuHostMemory& other) = delete;
	void operator=(const cuHostMemory&) = delete;

	static cuHostMemory* GetInstance(size_t min_size, int max_log_scale); 
	static void RemoveInstance();
	void* Malloc(size_t size);
	void Free(void* ptr);
	size_t GetTotalSize();
};

