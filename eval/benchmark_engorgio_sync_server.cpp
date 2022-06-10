#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "webmdec.h"
#include "controller.h"
#include "control_common.h"
#include "neural_enhancer.h"
// #include "test_common.h"
#include "tool_common.h"
#include "cxxopts.hpp"
#include "neural_enhancer_server.h"
#include "cuda_runtime.h"

using namespace std;

int main(int argc, char** argv)
{
    cxxopts::Options options("EngorgioBenchmark", "Measure the throughput of Engorgio");

    options.add_options()
    ("g,gpus", "Number of GPUs", cxxopts::value<int>())
    ("v,videos", "Number of videos", cxxopts::value<int>())
    ("i,instance", "Instance name", cxxopts::value<std::string>())
    ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // parameters
    int num_gpus = result["gpus"].as<int>();
    int num_streams = result["videos"].as<int>();
    std::string instance_name = result["instance"].as<std::string>();

    // gpu name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name = std::string(prop.name);
    std::replace(device_name.begin(), device_name.end(), ' ', '_');
    
    // // create profiles
    // // TODO: cpu pinning
    std::string log_subdir = "s" + std::to_string(num_streams) + "g" + std::to_string(num_gpus);
    std::filesystem::path log_dir = std::filesystem::path(ENGORGIO_RESULT_DIR) / "evaluation" / "engorgio" / device_name / log_subdir;
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();

    std::vector<int> decode_threads = get_decode_threads(instance_name, num_gpus);
    std::vector<std::vector<int>> infer_threads = get_infer_threads(instance_name, num_gpus);
    std::vector<int> encode_threads = get_encode_threads(instance_name, num_gpus);

    InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, 1, global_start, infer_threads, log_dir);
    JPEGProfile eprofile = JPEGProfile(80, 2, true, false, global_start, encode_threads, log_dir);

    
    // setup engines
    NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    NeuralEnhancerServer server("0.0.0.0:50051");
    server.LoadNeuralEnhancer(neural_enhancer);
    server.run();

    return 0;
}