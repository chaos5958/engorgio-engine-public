#include "cuHostMemoryV2.h"
#include <functional>
#include <vector>
#include <list>
#include <mutex>
#include <cuda_runtime.h>

// cuHostAlloc is used for pinned memory

cuHostMemoryV2* cuHostMemoryV2::pinstance_{ nullptr };
std::mutex cuHostMemoryV2::pmutex_;

std::vector<int> DEFAULT_RESOLUTIONS = {360, 720, 1080, 2160};
const int NUM_DEFAULT_FRAMES = 20;

size_t get_frame_size(int resolution)
{
    switch (resolution)
    {
    case 360:
        return 360 * 640 * 3;
        break;
    case 720:
        return 720 * 1280 * 3;
    case 1080:
        return 1080 * 1920 * 3;
        break;
    case 2160:
        return 2160 * 3840 * 3;
        break;
    default:
        throw std::invalid_argument("Invalid resolution");
        break;
    }
}

cuHostMemoryV2::cuHostMemoryV2(int num_gpus)
{
    cuFramePool *frame_pool;
    uint8_t *ptr;
    size_t size;
    
    for (auto resolution: DEFAULT_RESOLUTIONS)
    {
        frame_pool = new cuFramePool;
        frame_pool->num_frames = NUM_DEFAULT_FRAMES * num_gpus;
        frame_pools_[resolution] = frame_pool;
    
        size = get_frame_size(resolution);
        // std::cout << resolution << "," << size << std::endl;
        for (int i = 0; i < frame_pool->num_frames; i++)
        {
            if (cudaHostAlloc(&ptr, size, cudaHostAllocDefault) != cudaSuccess)
            {
                throw std::runtime_error("cudaHostAlloc failed");
            }
            frame_pools_[resolution]->frames.push_back(ptr);
        }
    }
}

void cuHostMemoryV2::RemoveInstance()
{
    std::lock_guard<std::mutex> lock(pmutex_);
    void *frame;

    if (pinstance_ != nullptr)
    {
        // free CUDA memory
        for (auto & [resolution, frame_pool]: pinstance_->frame_pools_)
        {
            // std::cout << resolution << "," << frame_pool->num_frames << "," << frame_pool->frames.size() << std::endl;
            while ((std::size_t) frame_pool->num_frames != frame_pool->frames.size())
            {}
 
            // std::cout << resolution << "," << frame_pool->frames.size() << std::endl;
            while (frame_pool->frames.size() > 0)
            {
                frame = frame_pool->frames.front();
                cudaFreeHost(frame);
                frame_pool->frames.pop_front();
            }
        }

        // free the singletone instance
        delete pinstance_;
        pinstance_ = nullptr;
    }
}

cuHostMemoryV2* cuHostMemoryV2::GetInstance(int num_gpus)
{
    std::lock_guard<std::mutex> lock(pmutex_);
    if (pinstance_ == nullptr)
    {
        pinstance_ = new cuHostMemoryV2(num_gpus);
    }
    return pinstance_;
}


cuHostMemoryV2* cuHostMemoryV2::GetInstance()
{
    std::lock_guard<std::mutex> lock(pmutex_);
    return pinstance_;
}

void* cuHostMemoryV2::Malloc(int resolution)
{
    void *buf, *ptr;
    cuFramePool *fpool = frame_pools_[resolution];
    
    // std::cout << "Malloc: " << fpool->frames.size() << std::endl;
    fpool->mutex.lock();
    if (fpool->frames.size() == 0)
    {       
        size_t size = get_frame_size(resolution);
        for (int i = 0; i < NUM_DEFAULT_FRAMES; i++)
        {
            if (cudaHostAlloc(&ptr, size, cudaHostAllocDefault) != cudaSuccess)
            {
                throw std::runtime_error("cudaHostAlloc failed");
            }
            frame_pools_[resolution]->frames.push_back(ptr);
        }
        frame_pools_[resolution]->num_frames += NUM_DEFAULT_FRAMES;
    }  
    buf = fpool->frames.front();
    fpool->frames.pop_front();
    frame_pools_[resolution]->mutex.unlock();

    return buf;
}

void cuHostMemoryV2::Free(int resolution, void* ptr)
{
    cuFramePool *fpool = frame_pools_[resolution];
    fpool->mutex.lock();
    fpool->frames.push_back(ptr);
    fpool->mutex.unlock();

}

int cuHostMemoryV2::NumFrames(int resolution)
{
    cuFramePool *fpool = frame_pools_[resolution];
    return fpool->num_frames;
}












