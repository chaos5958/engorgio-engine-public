#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "NvInfer.h"
#include "common.h"
#include "logger.h"
#include "nlohmann/json.hpp"

template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
using json = nlohmann::json;

int get_bitrate(int resolution)
{
    switch(resolution)
    {
        case 360:
            return 700;
        case 720:
            return 4125;
        default:
            throw std::invalid_argument("Invalid resolution");
    }
}

static std::string get_train_video_name(int resolution, int duration)
{
    std::string resolution_str = std::to_string(resolution) + "p";
    std::string bitrate_str = std::to_string(get_bitrate(resolution)) + "kbps";
    std::string duration_str = "d" + std::to_string(duration);

    std::string name = resolution_str + "_" + bitrate_str + "_" + duration_str + "_train" + ".webm";
    return name;
}

std::string get_video_name(int resolution, int duration)
{
    std::string resolution_str = std::to_string(resolution) + "p";
    std::string bitrate_str = std::to_string(get_bitrate(resolution)) + "kbps";
    std::string duration_str = "d" + std::to_string(duration);

    std::string name = resolution_str + "_" + bitrate_str + "_" + duration_str + "_test" + ".webm";
    return name;
}

std::string get_onnx_path(std::string content, std::string model, int resolution, int duration, std::string data_dir)
{
    std::string video = get_train_video_name(resolution, duration);
    std::string path = data_dir + "/" + content + "/" + "checkpoint" + "/" + video + "/" + model + "/" + model + ".onnx";
    return path;
}

std::string get_trt_path(std::string content, std::string model, int resolution, int duration, std::string data_dir)
{
    std::string video = get_train_video_name(resolution, duration);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name = std::string(prop.name);
    std::replace(device_name.begin(), device_name.end(), ' ', '_');
    
    std::string path = data_dir + "/" + content + "/" + "checkpoint" + "/" + video + "/" + model + "/" + device_name + "/" + model + ".plan";
    return path;
}

double get_trt_latency(std::string model, int resolution, int duration, std::string data_dir)
{
    std::string video = get_video_name(resolution, duration);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name = std::string(prop.name);
    std::replace(device_name.begin(), device_name.end(), ' ', '_');
    
    std::string path = data_dir + "/" + "evaluation" + "/" + device_name + "/" + model + "/" +  video + "/" + "infer_result.json";
    std::ifstream ifs(path);
    json jf = json::parse(ifs);
    int num_frames = jf["num frames"];
    double latency = jf["latency"];

    return latency * 1000 / num_frames;
}

void read_file(const std::string& file, void** buffer, size_t *size)
{
    std::ifstream stream(file, std::ifstream::ate | std::ifstream::binary);
    if (!stream.is_open())
    {
        return;
    }

    *size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    // std::cout << *size << std::endl;
    *buffer = (unsigned char*)malloc(*size);
    if (*buffer == nullptr)
        return;

    stream.read((char*)*buffer, *size);
    return;
}

nvinfer1::IHostMemory *load_engine(void *buf, size_t size)
{
    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        return nullptr;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buf, size), 
                            samplesCommon::InferDeleter());
    if (!engine)
    {
        return nullptr;
    }   

    auto serialized_engine = engine->serialize();
    return serialized_engine;
}

void create_dummy_frames(uint8_t **buf, size_t *input_size, int num_frames, int resolution)
{
    switch (resolution)
    {
    case 360:
        *input_size = 360 * 640 * 3;
        break;
    case 720:
        *input_size = 1280 * 720 * 3;
        break;
    default:
        throw std::invalid_argument("Invalid resolution");
    }

    for (int i = 0; i < num_frames; i++)
    {
        *(buf + i) = (uint8_t*) malloc(sizeof(uint8_t) * (*input_size) * num_frames);
    }
    
}

int get_img_size(int resolution)
{
    switch (resolution)
    {
    case 360:
        return 360 * 640 * 3;
        break;
    case 720:
        return 720 * 1280 * 3;
        break;
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

std::pair<int, int> get_img_dimensions(int resolution)
{
    switch (resolution)
    {
    case 360:
        return std::make_pair<int, int>(640, 360);
        break;
    case 720:
        return std::make_pair<int, int>(1280, 720);
        break;
    case 1080:
        return std::make_pair<int, int>(1920, 1080);
        break;
    case 2160:
        return std::make_pair<int, int>(3840, 2160);
        break;
    default:
        throw std::invalid_argument("Invalid resolution");
        break;
    }
}


int get_anchor_thread(std::string instance)
{
    int thread_idx;
    if(instance == std::string("mango1"))
    {
        thread_idx = 2;
    }
    else if(instance == std::string("ae_server"))
    {
        thread_idx = 2;
    }
    else if(instance == std::string("g4dn.12xlarge"))
    {
        thread_idx = 2;
    }
    else if(instance == std::string("p3.2xlarge"))
    {
        thread_idx = 0;
    }
    else if(instance == std::string("g5.2xlarge"))
    {
        thread_idx = 0;
    }
    return thread_idx;
}

std::vector<std::vector<int>> get_infer_threads(std::string instance, int num_gpus)
{
    std::vector<std::vector<int>> thread_indexes;
    if(instance == std::string("mango1"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes.push_back(std::vector<int>({4, 6, 8}));
                break;
            case 2:
                thread_indexes.push_back(std::vector<int>({4, 6, 8}));
                thread_indexes.push_back(std::vector<int>({10, 12, 14}));
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    if(instance == std::string("ae_server"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes.push_back(std::vector<int>({3, 4, 5}));
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("g4dn.12xlarge"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes.push_back(std::vector<int>({3, 4, 5}));
                break;
            case 2:
                thread_indexes.push_back(std::vector<int>({3, 4, 5}));
                thread_indexes.push_back(std::vector<int>({6, 7, 8}));
                break;
            case 4:
                thread_indexes.push_back(std::vector<int>({3, 4, 5}));
                thread_indexes.push_back(std::vector<int>({6, 7, 8}));
                thread_indexes.push_back(std::vector<int>({9, 10, 11}));
                thread_indexes.push_back(std::vector<int>({12, 13, 14}));
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    } 
    else if(instance == std::string("p3.2xlarge"))
        thread_indexes.push_back(std::vector<int>({1, 2}));
    else if(instance == std::string("g5.2xlarge"))
        thread_indexes.push_back(std::vector<int>({1, 2}));

    return thread_indexes;
}

std::vector<int> get_encode_threads(std::string instance, int num_gpus)
{
    std::vector<int> thread_indexes;
    if(instance == std::string("mango1"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({10, 12});
                break;
            case 2:
                thread_indexes = std::vector<int>({16, 18, 20, 22});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("ae_server"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({6, 7});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("g4dn.12xlarge"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({6, 7});
                break;
            case 2:
                thread_indexes = std::vector<int>({9, 10, 11, 12});
                break;
            case 4:
                thread_indexes = std::vector<int>({15, 16, 17, 18, 19, 20, 21, 22});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("p3.2xlarge"))
        thread_indexes = std::vector<int>({3, 4});
    else if(instance == std::string("g5.2xlarge"))
        thread_indexes = std::vector<int>({3, 4});  

    return thread_indexes;
}

std::vector<int> get_libvpx_threads(std::string instance, int num_gpus)
{
    std::vector<int> thread_indexes;
    if(instance == std::string("mango1"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("g4dn.12xlarge"))
    {
        switch(num_gpus){
            case 1:
                for (int i = 13; i < 31; i++)
                    thread_indexes.push_back(i);
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("p3.2xlarge"))
        thread_indexes = std::vector<int>({6, 7});
    else if(instance == std::string("g5.2xlarge"))
        thread_indexes = std::vector<int>({6, 7});

    return thread_indexes;
}

std::vector<int> get_decode_threads(std::string instance, int num_gpus)
{
    // std::cout << instance << "," << num_gpus << std::endl;
    std::vector<int> thread_indexes;
    if(instance == std::string("mango1"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({25, 26, 27, 28, 29, 30, 31});
                break;
            case 2:
                thread_indexes = std::vector<int>({25, 26, 27, 28, 29, 30, 31});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("ae_server"))
    {
        switch(num_gpus){
            case 1:
                thread_indexes = std::vector<int>({8,9,10,11,12,13,14,15});
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("g4dn.12xlarge"))
    {
        switch(num_gpus){
            case 1:
                for (int i = 8; i < 13; i++)
                    thread_indexes.push_back(i);
                break;
            case 2:
                for (int i = 13; i < 24; i++)
                    thread_indexes.push_back(i);
                break;
            case 4:
                for (int i = 23; i < 48; i++)
                    thread_indexes.push_back(i);
                break;
            default:
                throw std::runtime_error("Unsupported num_gpus");
        }
    }
    else if(instance == std::string("p3.2xlarge"))
        thread_indexes = std::vector<int>({5, 6, 7});
    else if(instance == std::string("g5.2xlarge"))
        thread_indexes = std::vector<int>({5, 6, 7});

    return thread_indexes;
}
