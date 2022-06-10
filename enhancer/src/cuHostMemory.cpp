#include "cuHostMemory.h"
#include <functional>
#include <vector>
#include <list>
#include <mutex>
#include <cuda_runtime.h>

// cuHostAlloc is used for pinned memory

cuHostMemory* cuHostMemory::pinstance_{ nullptr };
std::mutex cuHostMemory::pmutex_;

cuHostMemory::cuHostMemory(size_t min_size, int max_log_scale) : min_size_(min_size), max_log_scale_(max_log_scale), total_size_(0)
{
    if (min_size <= 0 || max_log_scale > 32 || max_log_scale < 0)
        throw std::runtime_error("Invalid arguments");

    fragments_.resize(64, nullptr); // hyunho: why uses 64 instead of 32 here?

    cudaMalloc();
}

bool cuHostMemory::cudaMalloc()
{
    int log_scale = max_log_scale_ + 1;
    size_t fragment_size = min_size_ + sizeof(FragmentHeader);
    uint8_t* ptr;
    FragmentHeader* curr_fragment,* prev_fragment;
    std::vector<FragmentHeader*> new_fragments;

    if (cudaHostAlloc(&ptr, fragment_size << log_scale, cudaHostAllocDefault) != cudaSuccess)
        return false;

    ptrs_.push_back(ptr);
    total_size_ += fragment_size << log_scale;
    curr_fragment = (FragmentHeader*)ptr;
    curr_fragment->prev = nullptr;
    curr_fragment->next = (FragmentHeader*)(ptr + fragment_size);
    curr_fragment->log_scale = 0;
    curr_fragment->divided = 0;
    curr_fragment->allocated = true;

    prev_fragment = curr_fragment;
    for (int i = 0; i < log_scale; i++)
    {
        curr_fragment = curr_fragment->next;
        curr_fragment->prev = prev_fragment;
        curr_fragment->next = (FragmentHeader*)((uint8_t*)curr_fragment + (fragment_size << i));
        curr_fragment->log_scale = i;
        curr_fragment->divided = 1;
        curr_fragment->allocated = false;
        prev_fragment = curr_fragment;
        new_fragments.push_back(curr_fragment);
    }
    curr_fragment->next = nullptr;

    // need a lock for thread-safety
    mutex_.lock();
    for (auto& fragment : new_fragments)
    {
        InsertFragment(fragments_, fragment);
    }
    mutex_.unlock();

    return true;
}

void cuHostMemory::RemoveInstance()
{
    std::lock_guard<std::mutex> lock(pmutex_);

    if (pinstance_ != nullptr)
    {
        // free CUDA memory
        for (auto& ptr: pinstance_->ptrs_)
        {
            cudaFreeHost(ptr);
        }

        // free the singletone instance
        delete pinstance_;
        pinstance_ = nullptr;
    }
}

cuHostMemory* cuHostMemory::GetInstance(size_t min_size, int max_log_scale)
{
    std::lock_guard<std::mutex> lock(pmutex_);
    if (pinstance_ == nullptr)
    {
        pinstance_ = new cuHostMemory(min_size, max_log_scale);
    }
    return pinstance_;
}

void* cuHostMemory::Malloc(size_t size)
{
    int log_scale, min_log_scale = 0;
    FragmentHeader* target_fragment, * curr_fragment, * prev_fragment, * next_fragment;
    size_t fragment_size = min_size_ + sizeof(FragmentHeader);

    if (size <= 0 || size > (size_t)min_size_ << max_log_scale_)
        return nullptr;

    while (size > ((size_t)min_size_ << min_log_scale))
        min_log_scale++;

    mutex_.lock();

    // case 1: optimal fragment exists
    if (fragments_[min_log_scale] != nullptr)
    {
        target_fragment = PopFragment(fragments_, min_log_scale);
        mutex_.unlock();
        return (uint8_t*)target_fragment + sizeof(FragmentHeader);
    }

    // TODO (½ÂÁØ): 1) <= or < which is correct?, 2) why selecting the last index (LOG_FRAGMENT)?
    bool found = false;
    while (!found)
    {
        for (log_scale = min_log_scale; log_scale <= max_log_scale_; log_scale++)
        {
            if (fragments_[log_scale] != nullptr)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            mutex_.unlock();
            if (!cudaMalloc())
                return nullptr;
            mutex_.lock();
        }
    }
    
    // case 2: sub-optimal fragment exists
    target_fragment = PopFragment(fragments_, log_scale);
    if (log_scale != min_log_scale)
    {
        next_fragment = target_fragment->next;
        target_fragment->next = (FragmentHeader*)((uint8_t*)target_fragment + (fragment_size << min_log_scale));
        target_fragment->log_scale = min_log_scale;
        target_fragment->divided <<= (log_scale - min_log_scale); // TODO (½ÂÁØ): understand this line (ok)

        prev_fragment = target_fragment;
        curr_fragment = target_fragment;
        for (int i = min_log_scale; i < log_scale; i++)
        {
            curr_fragment = curr_fragment->next;
            curr_fragment->prev = prev_fragment;
            curr_fragment->next = (FragmentHeader*)((uint8_t*)curr_fragment + (fragment_size << i));
            curr_fragment->log_scale = i;
            curr_fragment->divided = target_fragment->divided << (i - min_log_scale); // TODO (½ÂÁØ): understand this line (ok)
            curr_fragment->divided |= 1;
            InsertFragment(fragments_, curr_fragment);
            prev_fragment = curr_fragment;
        }

        curr_fragment->next = next_fragment;
        if (next_fragment != nullptr)
            next_fragment->prev = curr_fragment;
    }

    mutex_.unlock();
    return (uint8_t*)target_fragment + sizeof(FragmentHeader);
}

void cuHostMemory::Free(void* ptr)
{
    FragmentHeader* target_fragment, * next_fragment, * prev_fragment;
    int log_scale;

    if (ptr == nullptr)
        return;

    mutex_.lock();
    
    target_fragment = (FragmentHeader*)((uint8_t*)ptr - sizeof(FragmentHeader));
    log_scale = target_fragment->log_scale;
    next_fragment = target_fragment->next;

    while (true)
    {
        if (!(target_fragment->divided & 1) && next_fragment != nullptr
            && !next_fragment->allocated && next_fragment->log_scale == log_scale)
        {
            RemoveFragment(fragments_, next_fragment);
            next_fragment = next_fragment->next;
        }
        else if ((target_fragment->divided & 1) && (prev_fragment = target_fragment->prev) != nullptr
            && !prev_fragment->allocated && prev_fragment->log_scale == log_scale)
        {
            RemoveFragment(fragments_, prev_fragment);
            next_fragment = target_fragment->next;
            target_fragment = prev_fragment;
        }
        else
        {
            break;
        }
        
        target_fragment->next = next_fragment;
        target_fragment->divided >>= 1;
        log_scale++;
    }

    if (next_fragment != nullptr)
        next_fragment->prev = target_fragment;
    target_fragment->log_scale = log_scale;
    InsertFragment(fragments_, target_fragment);

    mutex_.unlock();
}

size_t cuHostMemory::GetTotalSize()
{
    return total_size_;
}












