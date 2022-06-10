#include <string>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>
#include <nvml.h>

#include "infer_engine.h"
#include "NvInfer.h"
#include "tool_common.h"

const int NUM_DUMMY_ITERS = 5;

void test_single_stream_async(InferEngineType type, std::vector<EngorgioModel*> onnx_model_per_request, EngorgioModel *trt_model, 
                std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
{
    // InferEngineProfile profile = InferEngineProfile(type, true);
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    auto deadline = global_start + std::chrono::seconds(4); // dummy
    std::vector<std::vector<int>> thread_indexes = {{4, 6, 8}};
    InferEngineProfile profile = InferEngineProfile(type, true, 1, global_start, thread_indexes);
    InferEngine *infer_engine = new InferEngine(profile);
    infer_engine->LoadEngine(trt_model);
    // std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    for (int i = 0; i < NUM_DUMMY_ITERS; i++)
    {
        infer_engine->EnhanceAsync(0, onnx_model_per_request[i], nullptr, frames_per_request[i], deadline);
    }
    while (!infer_engine->Finished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_repeats; i++)
    {
        infer_engine->EnhanceAsync(0, onnx_model_per_request[i+NUM_DUMMY_ITERS], nullptr, frames_per_request[i+NUM_DUMMY_ITERS], deadline);
    }
    while (!infer_engine->Finished())
    {
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (End-to-end): " << elapsed.count() * 1000 << "ms" << std::endl;

    // infer_engine->SaveLog();
    delete infer_engine;
}

// void test_single_stream_sync(InferEngineType type, InferModel *onnx_model, InferModel *trt_model, 
//                 InferFrames *frames, int num_repeats, int input_resolution, int output_resolution)
// {
//     InferEngineProfile profile = InferEngineProfile(type, true);
//     InferEngine *infer_engine = new InferEngine(profile);
//     infer_engine->LoadEngine(trt_model);
    
//     // dummy
//     for (int i = 0; i < NUM_DUMMY_ITERS; i++)
//     {
//         infer_engine->EnhanceSync(0, onnx_model, frames, input_resolution, output_resolution);
//     }

//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_repeats; i++)
//     {
//         infer_engine->EnhanceSync(0, onnx_model, frames, input_resolution, output_resolution);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Latency (End-to-end): " << elapsed.count() * 1000 << "ms" << std::endl;

//     infer_engine->Save();

//     delete infer_engine;
// }

int main()
{
    std::string content = "lol0";
    int resolution = 720;
    int duration = 600;
    std::pair<int, int> dimensions = get_img_dimensions(resolution);
    int scale = 3;
    int num_repeats = 100;
    int num_frames = 4;
    std::string model_name = "EDSR_B8_F32_S3";
    std::string onnx_path = get_onnx_path(content, model_name, resolution, duration);
    std::string trt_path = get_trt_path(content, model_name, resolution, duration);
    
    InferEngineType type;
    std::vector<EngorgioModel *> onnx_model_per_request;
    EngorgioModel *trt_model, *onnx_model;
    std::vector<std::vector<EngorgioFrame*>> frames_per_request;
    for (int i = 0; i < (num_repeats + NUM_DUMMY_ITERS); i++)
    {
        std::vector<EngorgioFrame*> frames;
        for (int j = 0; j < num_frames; j++)
        {
            frames.push_back(new EngorgioFrame(dimensions.first, dimensions.second));
        }
        frames_per_request.push_back(frames);
        onnx_model = new EngorgioModel(onnx_path, model_name, scale);
        onnx_model_per_request.push_back(onnx_model);
    }
    // return 0;
    type = InferEngineType::kEngorgio;
    
    trt_model = new EngorgioModel(trt_path, model_name, scale);
    test_single_stream_async(type, onnx_model_per_request, trt_model, frames_per_request, num_repeats);
    // test_single_stream_sync(type, onnx_model, trt_model, frames, num_repeats, input_resolution, output_resolution);

    // type = InferEngineType::kBaseline2;
    // test_single_stream_sync(type, onnx_model, trt_model, frames, num_repeats, input_resolution, output_resolution);
}