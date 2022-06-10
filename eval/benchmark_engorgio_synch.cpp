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
#include "remote_neural_enhancer.h"
#include "neural_enhancer.h"
// #include "test_common.h"
#include "tool_common.h"
#include "cxxopts.hpp"
#include "cuda_runtime.h"


using namespace std;

static bool FinishedEpochDecode(EngorgioStream* stream, int index, int epoch_length)
{
    bool finished = false;
    
    if (stream->frames.size() > 0)
    {
        EngorgioFrame *frame = stream->frames.back();
        if (frame != nullptr)
        {
            if (frame->current_video_frame == ((index + 1) * epoch_length - 1) && frame->frame_type != 1)
                finished = true;
        }        
    }

    return finished;
}

// TODO: validate step-by-step
int main(int argc, char** argv)
{
    cxxopts::Options options("EngorgioBenchmark", "Measure the throughput of Engorgio");

    options.add_options()
    ("g,gpus", "Number of GPUs", cxxopts::value<int>()->default_value("1"))
    ("v,videos", "Number of videos", cxxopts::value<int>()->default_value("1"))
    ("i,instance", "Instance name", cxxopts::value<std::string>()->default_value("mango1"))
    ("f,fraction", "Fraction of anchor frames", cxxopts::value<double>()->default_value("0.05"))
    ("e,epochs", "Number of epochs", cxxopts::value<int>()->default_value("1"))
    ("r,resolution", "Resolution", cxxopts::value<int>()->default_value("720"))
    // ("f,framerate", "Frame rate", cxxopts::value<int>()->default_value(60))
    ("d,duration", "Duration", cxxopts::value<int>()->default_value("600"))
    ("s,scale", "Scale", cxxopts::value<int>()->default_value("3"))
    ("m,model", "Model", cxxopts::value<std::string>()->default_value("EDSR_B8_F32_S3"))
    ("c,contents", "Contents", cxxopts::value<std::vector<std::string>>())
    ("x,server", "Server IP and Port", cxxopts::value<std::vector<std::string>>())
    ("skip-latency", "Skip Latency Load", cxxopts::value<bool>()->default_value("false"))
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
    int resolution = result["resolution"].as<int>();
    int duration = result["duration"].as<int>();
    double fraction = result["fraction"].as<double>();
    int scale = result["scale"].as<int>();
    int num_epochs = result["epochs"].as<int>();
    int epoch_length = 60;
    int gop = 120;
    std::string model_name = result["model"].as<std::string>();
    std::string instance_name = result["instance"].as<std::string>();
    std::vector<std::string> contents_list = result.count("contents") ? result["contents"].as<std::vector<std::string>>() : std::vector<std::string>();
    std::vector<std::string> server_ip_ports = result.count("server") ? result["server"].as<std::vector<std::string>>() : std::vector<std::string>();
    // std::pair<int, int> dimensions = get_img_dimensions(resolution);
    bool skip_latency = result["skip-latency"].as<bool>();
    int num_frames = num_epochs * epoch_length;

    // gpu name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name = std::string(prop.name);
    std::replace(device_name.begin(), device_name.end(), ' ', '_');

    // validation
    // std::cout << num_gpus << "," 
    //           << num_streams << "," 
    //           << resolution << "," 
    //           << duration << "," 
    //           << scale << "," 
    //           << num_epochs << "," 
    //           << model_name << "," 
    //           << num_frames << std::endl;
    
    // select contents
    // TODO: rollback
    std::vector<std::string> contents;
    if (contents_list.size() == 0) {
        int num_contents = ENGORGIO_CONTENTS.size();
        for (int i = 0; i < num_streams; i++)
        {
            contents.push_back(ENGORGIO_CONTENTS[rand() % num_contents]);
        }
    } else {
        while (contents.size() < num_streams) {
            for (auto it = contents_list.begin(); it != contents_list.end(); ++it) {
                contents.push_back(*it);
                if (contents.size() >= num_streams)
                    break;
            }
        }
    }

    // create models
    // 1. trt mdoel
    std::string trt_path = get_trt_path(contents[0], model_name, resolution, duration);
    EngorgioModel *trt_model = new EngorgioModel(trt_path, model_name, scale);
    // 2. per-stream onnx models
    std::string onnx_path;
    std::vector<EngorgioModel*> onnx_models;
    EngorgioModel *onnx_model;
    for (int i = 0; i < num_streams; i++)
    {
        onnx_path = get_onnx_path(contents[i], model_name, resolution, duration);
        onnx_model = new EngorgioModel(onnx_path, model_name, scale);
        onnx_models.push_back(onnx_model);
    }

    // create streams
    std::string video_name = get_video_name(resolution, duration);
    std::vector<VpxStream *> vpx_streams;
    VpxStream *vpx_stream;
    for (int i = 0; i < num_streams; i++)
    {
        vpx_stream = new VpxStream(contents[i], video_name, num_frames);
        vpx_streams.push_back(vpx_stream);
    }
    
    // // create profiles
    // // TODO: cpu pinning
    std::string log_subdir = "s" + std::to_string(num_streams) + "g" + std::to_string(num_gpus) + "f" + std::to_string(int(fraction*100));
    std::filesystem::path log_dir = std::filesystem::path(ENGORGIO_RESULT_DIR) / "evaluation" / "engorgio" / device_name / log_subdir;
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();

    std::vector<int> decode_threads = get_decode_threads(instance_name, num_gpus);
    int anchor_thread = get_anchor_thread(instance_name);
    std::vector<std::vector<int>> infer_threads = get_infer_threads(instance_name, num_gpus);
    std::vector<int> encode_threads = get_encode_threads(instance_name, num_gpus);

    DecodeEngineProfile dprofile = DecodeEngineProfile(decode_threads.size(), true, global_start, decode_threads, log_dir);
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_FRACTION, true, global_start, gop, fraction, anchor_thread, log_dir);
    InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, num_gpus, global_start, infer_threads, log_dir);
    JPEGProfile eprofile = JPEGProfile(80, encode_threads.size(), true, false, global_start, encode_threads, log_dir);

    // validation
    // std::cout << log_dir << std::endl;
    // std::cout << "anchor: " << anchor_thread << std::endl;
    // for (auto i : decode_threads)
    //     std::cout << "decode: " << i << std::endl;
    // for (auto threads : infer_threads)
    // {
    //     for (auto i : threads)
    //         std::cout << "infer: " << i << std::endl;
    // }
    // for (auto i : encode_threads)
    //     std::cout << "encode: " << i << std::endl;
    
    // setup engines
    Controller *controller = new Controller(dprofile, aprofile);
    NeuralEnhancer *neural_enhancer;
    if (server_ip_ports.size() > 0) {
        std::cerr << "Running in distributed mode." << std::endl;
        neural_enhancer = new RemoteNeuralEnhancer(iprofile, server_ip_ports);
    } else {
        std::cerr << "Running in local mode." << std::endl;
        neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    }
    controller->LoadNeuralEnhancer(neural_enhancer);
    if (!skip_latency) {
        double latency = get_trt_latency(model_name, resolution, duration);
        controller->LoadDNNLatency(resolution, model_name, latency);
    } else {
        std::cerr << "Skipping latency load." << std::endl;
    }
    neural_enhancer->LoadEngine(trt_model);
    
    // init streams
    std::vector<int> stream_ids;
    int stream_id;
    for (int i = 0; i < num_streams; i++)
    {
        stream_id = controller->Init(gop, contents[i], onnx_models[i]);
        stream_ids.push_back(stream_id);
    }

    // run streams
    // TODO: async processing
    EngorgioStream* stream;
    int idx;
    for (int i = 0; i < num_epochs; i++)
    {
        for (int j = 0; j < epoch_length; j++)
        {
            idx = i * epoch_length + j;
            for (int k = 0; k < num_streams; k++)
            {
                controller->Process(stream_ids[k], vpx_streams[k]->frames_[idx].second, vpx_streams[k]->frames_[idx].first);
            }
        }
        for (int k = 0; k < num_streams; k++)
        {
            stream = controller->GetStream(stream_ids[k]);
            while (!FinishedEpochDecode(stream, i, epoch_length))
            {}
        }
        controller->RunAnchorOnce();
    }
    
    // wait the decode engine finished
    for (int i = 0; i < num_streams; i++)
    {
        controller->Free(stream_ids[i]);
        while(!controller->FinishedDecode(stream_ids[i]))
        {}
        controller->DeleteFrames(stream_ids[i]);
    }
    std::cout << "finish the decode engine" << std::endl;

    // wait the anchor engine finished
    controller->Free();
    std::cout << "finish the anchor engine" << std::endl;
    int num_processed_frames = controller->GetTotalAnchors();

    // wait the neural enhancer finished
    while(!neural_enhancer->Finished(num_processed_frames))
    {}
    std::cout << "finish the neural enhnacer" << std::endl;

    // destroy all
    delete neural_enhancer;
    delete controller;
    for (int i = 0; i < num_streams; i++)
    {
        delete vpx_streams[i];
    }     
    return 0;
}