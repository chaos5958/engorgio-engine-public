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

using namespace std;
const int FRAME_INTERVAL = 1;

void run_single_stream(int gop, std::string &content, std::string &video_name, int limit)
{
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    VpxStream* vpx_stream = new VpxStream(content, video_name, limit);
    DecodeEngineProfile dprofile = DecodeEngineProfile(1, true, global_start);
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_BUDGET, gop);
    Controller *controller = new Controller(dprofile, aprofile);
    
    int stream_id = controller->Init(gop, content);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < limit; i++)
    {
        controller->Process(stream_id, vpx_stream->frames_[i].second, vpx_stream->frames_[i].first);
    }
    controller->Free(stream_id);
    while(!controller->FinishedDecode(stream_id))
    {}
    controller->DeleteFrames(stream_id);
    controller->Free();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Avg Latency (Decode): " << elapsed.count() * 1000 / limit << "ms" << std::endl;
    std::cout << "Avg Throughput (Decode): " << limit / elapsed.count() << "fps" << std::endl;
    
    delete controller;
    delete vpx_stream;
}

void run_multi_streams(int gop, std::string &content, std::string &video_name, int limit, int num_streams, int num_workers)
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
    std::cout << "Avg Latency (Decode): " << elapsed.count() * 1000 / (limit * num_streams) << "ms" << std::endl;
    std::cout << "Avg Throughput (Decode): " << (limit *num_streams) / elapsed.count() << "fps" << std::endl;

    delete controller;
    for (auto vpx_stream : vpx_streams)
        delete vpx_stream;
}

int main()
{
    ippInit();   

    std::string content = "lol0";
    int gop = 240;
    int resolution = 720;
    int duration = 600;
    int limit = 1000;
    std::string video_name = get_video_name(resolution, duration); 
    // run_single_stream(gop, content, video_name, limit);

    // multi-streams
    int num_streams = 5;
    int num_workers = 5;
    run_multi_streams(gop, content, video_name, limit, num_streams, num_workers);


    return 0;
}