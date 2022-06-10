#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <map>
#include <string>
#include "date/date.h"
#include "libvpx_engine.h"
#include "cuHostMemoryV2.h"
#include "./vpxenc_api.h"
#include "./vpx/vpx_codec.h"
#include "ipp.h"
#include "ippcc.h"
#include "ippi.h"

using namespace std;
namespace fs = std::filesystem;


void set_stream_params(struct VpxEncoderConfig *global,
                               struct stream_state *stream, int bitrate, bool save_video) {
    static const int *ctrl_args_map = vp9_arg_ctrl_map;
    struct stream_config *config = &stream->config;

  // Handle codec specific options
    config->out_fn = "/workspace/research/engorgio-engine/test.webm";
    // g_argv[2];//"/home/jaykim305/vpx-custom-enc/output/mylive_vp9.webm";
    // config->write_webm = 1;
    config->write_webm = 1;
    config->cfg.g_threads = 16;
    config->cfg.g_w = 3840;
    config->cfg.g_h = 2160;
    config->cfg.g_error_resilient = 1;
    config->cfg.rc_end_usage = VPX_CBR;
    config->cfg.g_lag_in_frames = 0;
    config->cfg.rc_dropframe_thresh = 0;
    config->cfg.rc_target_bitrate = bitrate;
    config->cfg.rc_min_quantizer = 4;
    config->cfg.rc_max_quantizer = 48;
    config->cfg.kf_min_dist = 0;
    config->cfg.kf_max_dist = 90;

    // codec ctrls
    set_arg_ctrl(config, ctrl_args_map, VP8E_SET_CPUUSED, 9);
    set_arg_ctrl(config, ctrl_args_map, VP8E_SET_STATIC_THRESHOLD, 0);
    set_arg_ctrl(config, ctrl_args_map, VP9E_SET_TILE_COLUMNS, 4);
    set_arg_ctrl(config, ctrl_args_map, VP9E_SET_FRAME_PARALLEL_DECODING, 1);
    set_arg_ctrl(config, ctrl_args_map, VP9E_SET_ROW_MT, 1);
    set_arg_ctrl(config, ctrl_args_map, VP8E_SET_MAX_INTRA_BITRATE_PCT, 300);
}

void set_global_config(struct VpxEncoderConfig *global) {
    memset(global, 0, sizeof(*global));
    global->codec = get_vpx_encoder_by_name("vp9");
    global->passes = 1;
    global->pass = 0;
    global->deadline = VPX_DL_REALTIME;
    global->color_type = I420;
    global->verbose = 1;
    global->have_framerate = 1;
    global->framerate.num = 60000; //60fps
    global->framerate.den = 1000;
    global->quiet = 1;
    global->show_psnr = 0;
}

libvpxEngine::libvpxEngine(libvpxProfile &profile)
{
    log_dir_ = profile.log_dir_;
    bitrate_ = profile.bitrate_;
    num_threads_ = profile.num_threads_;
    thread_indexes_ = profile.thread_indexes_;
    save_log_ = profile.save_log_;
    save_video_ = profile.save_video_;
    log_dir_ = profile.log_dir_;
    start_ = profile.start_;

    for (int i = 0; i < NUM_LIBVPX_STREAMS; i++)
        streams_.push_back(nullptr);
    encode_count_ = 0;

    cpu_set_t cpuset;
    encode_worker_ = new std::thread([this](){this->EncodeHandler(0);});
    CPU_ZERO(&cpuset);
    for (auto index : thread_indexes_)
        CPU_SET(index, &cpuset);
    int rc = pthread_setaffinity_np(encode_worker_->native_handle(),
                                sizeof(cpu_set_t), &cpuset);
    assert (rc == 0);
}

libvpxEngine::~libvpxEngine()
{
    for (auto &stream : streams_)
        delete stream;

    libvpxEvent event;
    event.type_ = libvpxEventType::kJoin;
    encode_mutex_.lock();
    encode_events_.push_back(event);
    encode_mutex_.unlock();
    encode_worker_->join();
}

void libvpxEngine::EncodeHandler(int index)
{
    libvpxEvent event;
    bool has_event = false;

    while (1)
    {
        encode_mutex_.lock();
        has_event = false;
        if (!encode_events_.empty())
        {
            event = encode_events_.front();
            encode_events_.pop_front();
            has_event = true;
        }
       encode_mutex_.unlock();

        if (has_event)
        {
            // std::cout << "unload event" << std::endl;
            switch (event.type_)
            {
            case libvpxEventType::kEncode:
                Encode(index, event);
                break;
            case libvpxEventType::kJoin:
                return;
            default:
                std::cerr << "Unsupported event type" << std::endl;
                break;
            }
        }
    }
}

void libvpxEngine::Encode(int index, libvpxEvent &event)
{
    int stream_id = event.stream_id_;
    std::vector<EngorgioFrame*> &frames = event.frames_;
    libvpxStream *stream = streams_[stream_id];
    std::chrono::system_clock::time_point start, end;
    libvpxFrameLog *log;

    int width = 3840;
    int height = 2160;
    int y_stride = width;
    int uv_stride = width / 2;
    int yuv_stride[3] = {y_stride, uv_stride, uv_stride};
    IppiSize roi_size = {width, height};
    int rgb_stride = width * 3;
    IppStatus st = ippStsNoErr;
    const int rgb_order[3] = {2, 1, 0};
    std::chrono::duration<double> elapsed;
    cuHostMemoryV2* host_memory = cuHostMemoryV2::GetInstance();

    int got_data;
    for (auto &frame : frames)
    {
        Ipp8u* rgb_buf = (Ipp8u*) frame->rgb_buffer;
        st = ippiSwapChannels_8u_C3R(rgb_buf, rgb_stride, rgb_buf, rgb_stride, roi_size, rgb_order);
        if ( st != ippStsNoErr)
        {
                std::cout << "failed: " << st << std::endl;
                return;
        }
        Ipp8u* yuv_buf[3] = {stream->raw.planes[0], stream->raw.planes[1], stream->raw.planes[2]};
        st = ippiBGRToYCbCr420_709CSC_8u_C3P3R(rgb_buf, rgb_stride, yuv_buf, yuv_stride, roi_size);
        assert (st == ippStsNoErr);
        // end = std::chrono::system_clock::now();
        // elapsed = end - start;
        // std::cout << elapsed.count() << std::endl;

        start = std::chrono::system_clock::now();
        got_data = 0;
        stream->frames_in++;
        get_cx_data(stream->stream, &stream->global, &got_data);
        encode_frame(stream->stream, &stream->global, &stream->raw, stream->frames_in);
        encode_count_ += 1;
        end = std::chrono::system_clock::now();
        elapsed = end - start;
        // std::cout << elapsed.count() << std::endl;

        log = new libvpxFrameLog(frame->current_video_frame, frame->current_video_frame, start, end);
        stream->logs.push_back(log);
    }

        // free sr frames
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        if (host_memory) // case 1: given by the infer engine
        {
            host_memory->Free(frames[i]->height, (void*)frames[i]->rgb_buffer);
            frames[i]->rgb_buffer = nullptr;
            delete frames[i];
        }
        else // case 2: given by the tester 
        {
            delete frames[i];
        }
    }

}

void libvpxEngine::SaveLog(int stream_id)
{
    if (!save_log_)
        return;

    // set base dir and create it 
    std::string today = date::format("%F", std::chrono::system_clock::now());
    std::filesystem::path log_dir;
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
        // log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "libvpx_engine" / today / std::to_string(stream_id);
        log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "libvpx_engine" / std::to_string(stream_id);
    }
    else
    {
        // log_dir = log_dir_ / today / std::to_string(stream_id);
        log_dir = log_dir_ / std::to_string(stream_id);
    }
    if (!fs::exists(log_dir))
        fs::create_directories(log_dir);
       
    // save logs
    fs::path latency_path;
    std::ofstream latency_file;
    std::string latency_log;

    libvpxStream *stream = streams_[stream_id];
    latency_path = log_dir / "encode_latency.txt";
    latency_file.open(latency_path);

    // std::cout << latency_path << std::endl;
    if (latency_file.is_open())
    {
        latency_file << "Video index\tSuper index\tStart(s)\tEnd(s)" << '\n';
        std::chrono::duration<double> start_elapsed, end_elapsed, latency;
        for (auto log : stream->logs)
        {
            start_elapsed = log->start - start_;
            end_elapsed = log->end - start_;
            latency = log->end - log->start;
            latency_file << std::to_string(log->video_index) << "\t"
                         << std::to_string(log->super_index) << "\t"
                         << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t"
                         << std::to_string(latency.count()) << "\n";
        }
    }
    latency_file.flush();
    latency_file.close();
}

bool libvpxEngine::EncodeFinished(int num_frames)
{
    if (encode_count_ == num_frames)
        return true;
    else
        return false;
}

void libvpxEngine::Init(int stream_id)
{
    libvpxStream *stream = new libvpxStream(stream_id, save_video_, bitrate_);
    streams_[stream_id] = stream;
}

void libvpxEngine::Encode(int stream_id, std::vector<EngorgioFrame*> &frames)
{
    libvpxEvent event = {libvpxEventType::kEncode, stream_id, frames};
    encode_mutex_.lock();
    encode_events_.push_back(event);
    encode_mutex_.unlock();
}

void libvpxEngine::Free(int stream_id)
{
    SaveLog(stream_id);
    delete streams_[stream_id];
    streams_[stream_id] = nullptr;
}