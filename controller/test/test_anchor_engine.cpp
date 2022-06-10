#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "webmdec.h"
#include "controller.h"
#include "control_common.h"
#include "neural_enhancer.h"
// #include "test_common.h"
#include "tool_common.h"

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

// single stream: N frames (decode) -> N frame (select) -> ...
void single_stream_synch_controller(int gop, std::string &content, int resolution, int duration, std::string &model_name, int scale, int num_epochs, int epoch_length, SelectPolicy policy)
{
    std::string video_name = get_video_name(resolution, duration);
    double latency = get_trt_latency(model_name, resolution, duration);
    std::string onnx_path = get_onnx_path(content, model_name, resolution, duration);
    EngorgioModel *onnx_model = new EngorgioModel(onnx_path, model_name, scale);
 
    // VpxStream *vpx_stream = load_stream(content, resolution, num_epochs * epoch_length);
    VpxStream *vpx_stream = new VpxStream(content, video_name, num_epochs * epoch_length);
    vector<pair<int, uint8_t*>> &video_frames = vpx_stream->frames_;
    
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    DecodeEngineProfile dprofile = DecodeEngineProfile(1, true, global_start, {0});
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_FRACTION, true, global_start, gop, 0.075, 2);
     InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, false, 1);
    // InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, 1, global_start, {{4, 6, 8}});
    
    Controller *controller = new Controller(dprofile, aprofile);
    NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile);
    controller->LoadNeuralEnhancer(neural_enhancer);
    controller->LoadDNNLatency(resolution, model_name, latency);
    // controller->RunAnchorPeriodic();
    
    int stream_id = controller->Init(gop, content, onnx_model);
    EngorgioStream* stream = controller->GetStream(stream_id);
    int idx;
    for (int i = 0; i < num_epochs; i++)
    {
        for (int j = 0; j < epoch_length; j++)
        {
            idx = i * epoch_length + j;
            controller->Process(stream_id, video_frames[idx].second, video_frames[idx].first);
        }
        while (!FinishedEpochDecode(stream, i, epoch_length))
        {}
        controller->RunAnchorOnce();
        controller->DeleteAnchors(stream_id);
    }
    controller->Free(stream_id);
    while(!controller->FinishedDecode(stream_id))
    {}
    controller->DeleteFrames(stream_id);
    controller->Free();
    delete neural_enhancer;
    delete controller;
    delete vpx_stream;
}

void single_stream_synch_e2e(int gop, std::string &content, int resolution, int duration, std::string &model_name, int scale, int num_epochs, int epoch_length, SelectPolicy policy)
{
    std::string video_name = get_video_name(resolution, duration);
    // double latency = get_trt_latency(model_name, resolution, duration);
    std::string trt_path = get_trt_path(content, model_name, resolution, duration);
    std::string onnx_path = get_onnx_path(content, model_name, resolution, duration);
    EngorgioModel *onnx_model = new EngorgioModel(onnx_path, model_name, scale);
    // EngorgioModel *trt_model = new EngorgioModel(trt_path, model_name, scale);
 
    // VpxStream *vpx_stream = load_stream(content, resolution, num_epochs * epoch_length);
    VpxStream *vpx_stream = new VpxStream(content, video_name, num_epochs * epoch_length);
    vector<pair<int, uint8_t*>> &video_frames = vpx_stream->frames_;
    
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    DecodeEngineProfile dprofile = DecodeEngineProfile(1, true, global_start, {0});
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_FRACTION, true, global_start, gop, 0.075, 2);
    //  InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, false, 1);
    // InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, 1, global_start, {{4, 6, 8}});
    // JPEGProfile eprofile = JPEGProfile(80, 2, true, true, global_start, {10, 12, 14});
    
    Controller *controller = new Controller(dprofile, aprofile);
    // NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    // controller->LoadNeuralEnhancer(neural_enhancer);
    // controller->LoadDNNLatency(resolution, model_name, latency);
    // neural_enhancer->LoadEngine(trt_model);
    // controller->RunAnchorPeriodic();
    
    int stream_id = controller->Init(gop, content, onnx_model);
    EngorgioStream* stream = controller->GetStream(stream_id);
    int idx;
    for (int i = 0; i < num_epochs; i++)
    {
        for (int j = 0; j < epoch_length; j++)
        {
            idx = i * epoch_length + j;
            controller->Process(stream_id, video_frames[idx].second, video_frames[idx].first);
        }
        while (!FinishedEpochDecode(stream, i, epoch_length))
        {}
        controller->RunAnchorOnce();
    }
    
    // wait the decode engine finished
    controller->Free(stream_id);
    while(!controller->FinishedDecode(stream_id))
    {}
    controller->DeleteFrames(stream_id);
    std::cout << "finish the decode engine" << std::endl;

    // wait the anchor engine finished
    controller->Free();
    std::cout << "finish the anchor engine" << std::endl;
    
    // int num_frames = controller->GetTotalAnchors();
    // wait the neural enhancer finished
    // while(!neural_enhancer->Finished(num_frames))
    // {}
    // std::cout << "finish the neural enhnacer" << std::endl;

    // destroy all
    // delete neural_enhancer;
    delete controller;
    delete vpx_stream;
}

void multi_stream_synch_e2e(int num_streams, int gop, std::string &content, int resolution, int duration, std::string &model_name, int scale, int num_epochs, int epoch_length, SelectPolicy policy)
{
    // create models
    std::string video_name = get_video_name(resolution, duration);
    double latency = get_trt_latency(model_name, resolution, duration);
    std::string trt_path = get_trt_path(content, model_name, resolution, duration);
    std::string onnx_path = get_onnx_path(content, model_name, resolution, duration);
    EngorgioModel *trt_model = new EngorgioModel(trt_path, model_name, scale);
    std::vector<EngorgioModel*> onnx_models;
    EngorgioModel *onnx_model;
    for (int i = 0; i < num_streams; i++)
    {
        onnx_model = new EngorgioModel(onnx_path, model_name, scale);
        onnx_models.push_back(onnx_model);
    }

    // create streams
    std::vector<VpxStream *> vpx_streams;
    VpxStream *vpx_stream;
    for (int i = 0; i < num_streams; i++)
    {
        vpx_stream = new VpxStream(content, video_name, num_epochs * epoch_length);
        vpx_streams.push_back(vpx_stream);
    }
    
    // create profiles
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    DecodeEngineProfile dprofile = DecodeEngineProfile(1, true, global_start, {0});
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_FRACTION, true, global_start, gop, 0.075, 2);
    // InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, 1, global_start, {{4, 6, 8}});
    // JPEGProfile eprofile = JPEGProfile(80, 2, true, true, global_start, {10, 12, 14});
    
    // setup engines
    Controller *controller = new Controller(dprofile, aprofile);
    // NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    // controller->LoadNeuralEnhancer(neural_enhancer);
    // controller->LoadDNNLatency(resolution, model_name, latency);
    // neural_enhancer->LoadEngine(trt_model);
    
    // run engines
    std::vector<int> stream_ids;
    int stream_id;
    for (int i = 0; i < num_streams; i++)
    {
        stream_id = controller->Init(gop, content, onnx_models[i]);
        stream_ids.push_back(stream_id);
    }

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

    // wait the neural enhancer finished
    // int num_frames = controller->GetTotalAnchors();
    // while(!neural_enhancer->Finished(num_frames))
    // {}
    // std::cout << "finish the neural enhnacer" << std::endl;

    // destroy all
    // delete neural_enhancer;
    delete controller;
    for (int i = 0; i < num_streams; i++)
    {
        delete vpx_streams[i];
    }    
}

// multi-stream: N * M frames (decode) -> N * M frame (select) -> ...
void multi_stream_synch(int gop, std::string &content, std::string &video_name, int num_epochs, int epoch_length, int num_streams, SelectPolicy policy)
{
    std::vector<VpxStream*> vpx_streams;
    std::vector<EngorgioStream*> streams;
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    DecodeEngineProfile dprofile = DecodeEngineProfile(1, true, global_start);    
    AnchorEngineProfile aprofile = AnchorEngineProfile(SelectPolicy::ENGORGIO_BUDGET, gop);
    
    VpxStream* vpx_stream;
    for (int i = 0; i < num_streams; i++)
    {
        vpx_stream = new VpxStream(content, video_name, num_epochs * epoch_length);
        vpx_streams.push_back(vpx_stream);
    }

    Controller *controller = new Controller(dprofile, aprofile);    
    int stream_id;
    for (auto vpx_stream : vpx_streams)
    {
        stream_id = controller->Init(gop, content);
        vpx_stream->stream_id_ = stream_id;
        streams.push_back(controller->GetStream(stream_id));
    }
    
    int idx;
    for (int i = 0; i < num_epochs; i++)
    {
        // send frames
        for (int j = 0; j < epoch_length; j++)
        {
            for (auto vpx_stream : vpx_streams)
            {
                idx = i * epoch_length + j;
                controller->Process(vpx_stream->stream_id_, vpx_stream->frames_[idx].second, vpx_stream->frames_[idx].first);
            }
        }

        // check decoding is finished
        for (auto stream: streams)
        {
            while (!FinishedEpochDecode(stream, i, epoch_length))
            {}
        }
        
        // run anchor selection
        controller->RunAnchorOnce();
        for (auto vpx_stream: vpx_streams)
        {
            controller->DeleteAnchors(vpx_stream->stream_id_);
        }
    }

    for (auto vpx_stream: vpx_streams)
    {
        controller->Free(vpx_stream->stream_id_);
        while(!controller->FinishedDecode(vpx_stream->stream_id_))
        {}
        controller->DeleteFrames(vpx_stream->stream_id_);
        delete vpx_stream;
    }
    delete controller;
}


int main()
{
    std::string content = "lol0";
    int resolution = 720;
    int duration = 600;
    int gop = 240;
    std::string video_name = get_video_name(resolution, duration);
    int num_epochs = 1;
    int epoch_length = 40;
    SelectPolicy policy = SelectPolicy::ENGORGIO_BUDGET;

    std::string model_name = "EDSR_B8_F32_S3";
    int scale = 3;
    int num_streams = 10;
    // single_stream_synch_e2e(gop, content, resolution, duration, model_name, scale, num_epochs, epoch_length, policy);
    multi_stream_synch_e2e(num_streams, gop, content, resolution, duration, model_name, scale, num_epochs, epoch_length, policy);
    // single_stream_synch_controller(gop, content, resolution, duration, model_name, scale, num_epochs, epoch_length, policy);

    // multi-streams
    // int num_streams = 40;
    // multi_stream_synch(gop, content, video_name, num_epochs, epoch_length, num_streams, policy);
    
    return 0;
}