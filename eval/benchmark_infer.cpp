#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <nvml.h>
#include <filesystem>
#include "cxxopts.hpp"
#include "nlohmann/json.hpp"
#include "infer_engine.h"
#include "NvInfer.h"
#include "tool_common.h"

const int NUM_DUMMY_ITERS = 10;
namespace fs = std::filesystem;
using json = nlohmann::json;

double test_single_stream_async(InferEngineType type, std::vector<EngorgioModel*> &onnx_model_per_request, EngorgioModel *trt_model, 
                 std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
{
    InferEngineProfile profile = InferEngineProfile(type, false, 1);
    InferEngine *infer_engine = new InferEngine(profile);
    infer_engine->LoadEngine(trt_model);
    
    // dummy
    for (int i = 0; i < NUM_DUMMY_ITERS; i++)
    {
        infer_engine->EnhanceAsync(0,  onnx_model_per_request[i], nullptr, frames_per_request[i]);
    }
    while (!infer_engine->Finished())
    {
        // std::cout << "here" << std::end`l;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_repeats; i++)
    {
        infer_engine->EnhanceAsync(0, onnx_model_per_request[i + NUM_DUMMY_ITERS], nullptr, frames_per_request[i + NUM_DUMMY_ITERS]);
    }
    while (!infer_engine->Finished())
    {
        // std::cout << "here" << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (End-to-end): " << elapsed.count() * 1000 << "ms" << std::endl;

    // infer_engine->SaveLog();
    delete infer_engine;

    return elapsed.count();
}

int main(int argc, char** argv)
{
    cxxopts::Options options("InferBenchmark", "Measure the throughput of neural inference");

    options.add_options()
    ("c,content", "Content", cxxopts::value<std::string>()->default_value("lol0"))
    ("r,resolution", "Resolution", cxxopts::value<int>()->default_value("720"))
    ("d,duration", "Duration", cxxopts::value<int>()->default_value("600"))
    ("s,scale", "Scale", cxxopts::value<int>()->default_value("3"))
    ("m,model", "Model", cxxopts::value<std::string>())
    ("i,instance", "Instance", cxxopts::value<std::string>())
    ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string content = result["content"].as<std::string>();
    int resolution = result["resolution"].as<int>();
    int duration = result["duration"].as<int>();
    std::pair<int, int> dimensions = get_img_dimensions(resolution);
    int scale = result["scale"].as<int>();
    std::string model_name = result["model"].as<std::string>();
    std::string instance_name = result["instance"].as<std::string>();
    std::string video_name = get_video_name(resolution, duration);

    int num_repeats = 100;
    int num_frames = 6;

    std::string onnx_path = get_onnx_path(content, model_name, resolution, duration);
    std::string trt_path = get_trt_path(content, model_name, resolution, duration);
    std::vector<EngorgioModel *> onnx_model_per_request;
    EngorgioModel *onnx_model;
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        onnx_model = new EngorgioModel(onnx_path, model_name, scale);
        onnx_model_per_request.push_back(onnx_model);
    }
    EngorgioModel *trt_model = new EngorgioModel(trt_path, model_name, scale);
    std::vector<std::vector<EngorgioFrame*>> frames_per_request;
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        std::vector<EngorgioFrame*> frames;
        for (int j = 1; j <= num_frames; j++)
            frames.push_back(new EngorgioFrame(dimensions.first, dimensions.second));
        frames_per_request.push_back(frames);
    }

    double latency = test_single_stream_async(InferEngineType::kEngorgio, onnx_model_per_request, trt_model, frames_per_request, num_repeats);
    double throughput = (num_frames * num_repeats) / latency;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name = std::string(prop.name);
    std::replace(device_name.begin(), device_name.end(), ' ', '_');

    std::string resolution_name = std::to_string(resolution) + "p";
    fs::path json_dir = fs::path(ENGORGIO_RESULT_DIR) / "evaluation" / instance_name / resolution_name / model_name;
    if (!fs::exists(json_dir))
        fs::create_directories(json_dir);
    fs::path json_path = json_dir / "infer_result.json";

    json object;
    object["num frames"] = num_repeats * num_frames;
    object["latency"] = latency;
    object["throughput"] = throughput;

    std::ofstream json_file(json_path);
    json_file << std::setw(4) << object << std::endl;

    return 0;
}
