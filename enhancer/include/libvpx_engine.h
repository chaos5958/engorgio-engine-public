#pragma once

#include <cassert>
#include <thread>
#include <vector>
#include <turbojpeg.h>
#include <filesystem>
#include "control_common.h"
#include "vpxenc_api.h"
#include "./vpx/vpx_codec.h"

const int NUM_LIBVPX_STREAMS = 100;

void set_stream_params(struct VpxEncoderConfig *global,
                               struct stream_state *stream, int bitrate, bool save_video);

void set_global_config(struct VpxEncoderConfig *global);

enum class libvpxEventType : int
{
	kEncode = 0,
    kJoin = 1,
};


struct libvpxEvent
{
    libvpxEventType type_;
    int stream_id_;
    std::vector<EngorgioFrame*> frames_;
};

struct libvpxProfile{
    int bitrate_; // mbps
    int num_threads_;
    std::filesystem::path log_dir_;
    bool save_log_, save_video_;
    std::chrono::system_clock::time_point start_;
    std::vector<int> thread_indexes_;

    libvpxProfile(int bitrate, int num_threads, std::vector<int> thread_indexes = {})
    {
        bitrate_ = bitrate;
        num_threads_ = num_threads;
        save_log_ = false;
        
        if (thread_indexes.size() == 0)
        {
            for (int i = 0; i < num_threads_ + 1; i++)
            {
                thread_indexes_.push_back(i);
            }
        }
        else
        {
            // assert(thread_indexes.size() == num_threads_);
            thread_indexes_ = thread_indexes;
        }
    }

    libvpxProfile(int bitrate, int num_threads, bool save_log, bool save_video, std::chrono::system_clock::time_point start, std::vector<int> thread_indexes = {}, std::filesystem::path save_dir=std::filesystem::path(""))
    {
        bitrate_ = bitrate;
        num_threads_ = num_threads;
        save_log_ = save_log;
        save_video_ = save_video;
        start_ = start;
        log_dir_ = save_dir;

        if (thread_indexes.size() == 0)
        {
            for (int i = 0; i < num_threads_ + 1; i++)
            {
                thread_indexes_.push_back(i);
            }
        }
        else
        {
            // assert(thread_indexes.size() == num_threads_);
            thread_indexes_ = thread_indexes;
        }
    }
};

struct libvpxFrameLog{
    int video_index, super_index;
    std::chrono::system_clock::time_point start, end;

    libvpxFrameLog(int video_index_, int super_index_, std::chrono::system_clock::time_point start_, std::chrono::system_clock::time_point end_)
    {
        video_index = video_index_;
        super_index = super_index_;
        start = start_;
        end = end_;
    }
};

struct libvpxStream {
    int stream_id;
    int frames_in;
    vpx_image_t raw;
    struct VpxEncoderConfig global;
    bool save_video;
    struct stream_state *stream = NULL;
    std::vector<libvpxFrameLog*> logs;

    libvpxStream(int stream_id_, bool save_video_, int bitrate_)
    {
        frames_in = 0;
        stream_id = stream_id_;
        save_video = save_video_;
        memset(&raw, 0, sizeof(raw)); 

        // codec configuration 
        set_global_config(&global);
        stream = new_stream(&global, NULL);
        set_stream_params(&global, stream, bitrate_, save_video_);
        // std::cout << stream->config.cfg.g_w << "," << stream->config.cfg.g_h << std::endl;
        set_stream_dimensions(stream, stream->config.cfg.g_w, stream->config.cfg.g_h);
        validate_stream_config(stream, &global);
        struct VpxRational pixel_aspect_ratio = {1, 1};
        assert(vpx_img_alloc(&raw, VPX_IMG_FMT_I420, stream->config.cfg.g_w, stream->config.cfg.g_h, 32) != NULL);
        open_output_file(stream, &global, &pixel_aspect_ratio);

        initialize_encoder(stream, &global);
        // std::cout << "init a codec" << std::endl;
    }

    ~libvpxStream()
    {
        // std::cout << "here" << std::endl;
        vpx_codec_destroy(&stream->encoder);
        close_output_file(stream, global.codec->fourcc);
        // std::cout << "here1" << std::endl;

        vpx_img_free(&raw);
        free(stream);

        for (auto &log : logs)
        {
            delete log;
        }
        // std::cout << "here2" << std::endl;
    }
};

class libvpxEngine{
private:
    int bitrate_; // mbps
    int num_threads_;
    std::vector<int> thread_indexes_;
    std::vector<libvpxStream*> streams_;

    // logging
    bool save_log_, save_video_;
    std::filesystem::path log_dir_;
    std::chrono::system_clock::time_point start_;

    // debugging
    std::mutex encode_count_mutex_;
    int encode_count_;

    std::thread* encode_worker_;
    std::deque<libvpxEvent> encode_events_;
    std::mutex encode_mutex_;

    void EncodeHandler(int index);
    void Encode(int index, libvpxEvent &event);
    void SaveLog(int stream_id);

public:
    libvpxEngine(libvpxProfile &profile);
    ~libvpxEngine();

    bool EncodeFinished(int num_frames);
    void Init(int stream_id);
    void Encode(int stream_id, std::vector<EngorgioFrame*> &frames);
    void Free(int stream_id);
};