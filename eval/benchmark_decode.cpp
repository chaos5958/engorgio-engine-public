#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "webmdec.h"
#include "controller.h"
#include "control_common.h"
// #include "test_common.h"
#include "neural_enhancer.h"
#include "tool_common.h"
#include "ipp.h"
#include "cxxopts.hpp"
#include "nlohmann/json.hpp"


using namespace std;
const int FRAME_INTERVAL = 1;
namespace fs = std::filesystem;
using json = nlohmann::json;


std::vector<double> run_multi_streams(int gop, std::string &content, std::string &video_name, int limit, int num_streams, int num_workers)
{
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    std::vector<VpxStream*> vpx_streams;
    VpxStream* vpx_stream;
    for (int i = 0; i < num_streams; i++)
    {
        vpx_stream = new VpxStream(content, video_name, limit);
        vpx_streams.push_back(vpx_stream);
    }
    DecodeEngineProfile dprofile = DecodeEngineProfile(num_workers, true, global_start, {0, 2, 4, 6, 8});
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_BUDGET, gop);
    Controller *controller = new Controller(dprofile, aprofile);
    
    int stream_id;
    for (auto vpx_stream : vpx_streams)
    {
        stream_id = controller->Init(gop, content);
        vpx_stream->stream_id_ = stream_id;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < limit; i++)
    {
        for (auto vpx_stream : vpx_streams)
            controller->Process(vpx_stream->stream_id_, vpx_stream->frames_[i].second, vpx_stream->frames_[i].first);
    }
    for (auto vpx_stream: vpx_streams)
        controller->Free(vpx_stream->stream_id_);
    for (auto vpx_stream: vpx_streams)
    {
        while(!controller->FinishedDecode(vpx_stream->stream_id_))
        {}
    }
    for (auto vpx_stream: vpx_streams)
        controller->DeleteFrames(vpx_stream->stream_id_);
    controller->Free();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Avg Latency (Decode): " << elapsed.count() * 1000 / (limit * num_streams) << "ms" << std::endl;
    // std::cout << "Avg Throughput (Decode): " << (limit *num_streams) / elapsed.count() << "fps" << std::endl;
    std::cout << "Avg Latency (Decode): " << elapsed.count() * 1000 / limit << "ms" << std::endl;
    std::cout << "Avg Throughput (Decode): " << limit / elapsed.count() << "fps" << std::endl;

    delete controller;
    for (auto vpx_stream : vpx_streams)
        delete vpx_stream;

    std::vector<double> results;
    results.push_back(elapsed.count() * 1000);
    results.push_back(limit);
    results.push_back(limit / elapsed.count());
    return results;
}

int main(int argc, char** argv)
{
    cxxopts::Options options("InferBenchmark", "Measure the throughput of neural inference");
    options.add_options()
    ("c,content", "Content", cxxopts::value<std::string>()->default_value("lol0"))
    ("r,resolution", "Resolution", cxxopts::value<int>()->default_value("720"))
    ("d,duration", "Duration", cxxopts::value<int>()->default_value("600"))
    ("s,scale", "Scale", cxxopts::value<int>()->default_value("3"))
    ("l,limit", "Limit", cxxopts::value<int>()->default_value("600")) //TODO: use it 3600
    ("i,instance", "Instance", cxxopts::value<std::string>())
    ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    ippInit();   

    std::string content = result["content"].as<std::string>();
    int resolution = result["resolution"].as<int>();
    int duration = result["duration"].as<int>();
    int limit = result["limit"].as<int>();
    std::string instance_name = result["instance"].as<std::string>();
    std::string video_name = get_video_name(resolution, duration);

    // multi-streams
    int num_streams = 1;
    int num_workers = 1;
    int gop = 120;
    std::vector<double> curr_results, prev_results;
    
    // TODO: save a prev result
    while(1)
    {
        std::vector<double> curr_results = run_multi_streams(gop, content, video_name, limit, num_streams, num_workers);
        if (curr_results[2] < 60.0)
            break;
        num_streams += 1;
    }
    num_streams -= 1;
    
    // Save a json file
    std::string resolution_name = std::to_string(resolution) + "p";
    fs::path json_dir = fs::path(ENGORGIO_RESULT_DIR) / "evaluation" / instance_name / resolution_name;
    if (!fs::exists(json_dir))
        fs::create_directories(json_dir);
    fs::path json_path = json_dir / "decode_result.json";

    json object;
    object["num_streams"] = num_streams;

    std::ofstream json_file(json_path);
    json_file << std::setw(4) << object << std::endl;

    return 0;
}