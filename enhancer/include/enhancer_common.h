#pragma once 

#include <vector>
#include <filesystem>
#include <thread>
#include <mutex>
#include <deque>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuDeviceMemory.h"
#include "cuHostMemoryV2.h"
#include "common.h"

class EngorgioFramePool;
class EngorgioFrame;

using buffers_t = std::pair <void*, void*>; // input, output buffer pair

// TODO: add a scale factor
struct EngorgioModel
{
    void *buf_;
    size_t size_;
    std::string name_;
    int scale_;

    EngorgioModel(void *buf, size_t size, std::string name, int scale)
    {
        buf_ = buf;
        size_ = size;
        name_ = name;
        scale_ = scale;
    }

    EngorgioModel(std::string &path, std::string name, int scale)
    {
        name_ = name;
        scale_ = scale;

        std::ifstream stream(path, std::ifstream::ate | std::ifstream::binary);
        if (!stream.is_open())
        {
            throw std::invalid_argument("File open failed");
        }

        size_ = stream.tellg();
        stream.seekg(0, std::ios::beg);
        // std::cout << *size << std::endl;
        buf_ = (unsigned char*)malloc(size_);
        if (buf_ == nullptr)
        {
            throw std::runtime_error("Malloc failed");
        }
        stream.read((char*)buf_, size_);
    }

    ~EngorgioModel()
    {
        if (buf_)
            free(buf_);
    }
};