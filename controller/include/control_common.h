#pragma once

#include <iostream>
#include <list> 
#include <vector>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"

class EngorgioModel;

const int INIT_NUM_FRAMES = 200;


struct EngorgioFrame {
    // frame raw data
    uint8_t *rgb_buffer;
    int width, height;
    int width_alloc, height_alloc;

    // frame metadata
    int frame_type;
    int residual_size;
    int current_video_frame, current_super_frame;

    // anchor engine
    int stream_id;
    bool is_sorted = false;
    double diff_residual = 0;
    int prev_total_residual = 0;
    int offset = 0;

    // log
    std::chrono::system_clock::time_point decode_start, decode_end;

    EngorgioFrame()
    {
        rgb_buffer = nullptr;
        is_sorted = false;
        diff_residual = 0;
        prev_total_residual = 0;
        offset = 0;
    }
    // for testing the neural enhancer
    EngorgioFrame (int width_, int height_)
    {
        current_video_frame = 0;
        current_super_frame = 0;
        width = width_;
        height = height_;
        rgb_buffer = (uint8_t *) malloc(sizeof(uint8_t) * width * height * 3);
    }
    ~EngorgioFrame()
    {
        if (rgb_buffer)
            free(rgb_buffer);
    }
};

struct EngorgioFramePool
{
    std::mutex mutex;
    std::deque<EngorgioFrame*> all_frames;
    std::deque<EngorgioFrame*> free_frames;

    EngorgioFramePool(int num_frames, int width, int height)
    {
        EngorgioFrame *frame;
        for (int i = 0; i < num_frames; i++)
        {
            frame = new EngorgioFrame();
            frame->rgb_buffer = (uint8_t*) malloc(sizeof(uint8_t) * width * height * 3);
            frame->width_alloc = width;
            frame->height_alloc = height;
            all_frames.push_back(frame);
            free_frames.push_back(frame);
        }
    };

    ~EngorgioFramePool()
    {
        while(all_frames.size() != free_frames.size())
        {    
        }
        // TODO: validate this
        for (auto f : all_frames)
        {
            if (f)
                delete f;
        }
    };

    int Size()
    {
        return all_frames.size();
    };

    EngorgioFrame* GetFrame(int width, int height)
    {
        EngorgioFrame *frame;
        mutex.lock();
        if (free_frames.size() == 0)
        {
            frame = new EngorgioFrame();
            frame->rgb_buffer = (uint8_t*) malloc(sizeof(uint8_t) * width * height * 3);
            frame->width_alloc = width;
            frame->height_alloc = height;
            frame->width = width;
            frame->height = height;
            all_frames.push_back(frame);
        }
        else
        {
            frame = free_frames.front();
            free_frames.pop_front();
            if (width * height > frame->width_alloc * frame->height_alloc)
            {
                frame->rgb_buffer = (uint8_t *) realloc(frame->rgb_buffer, sizeof(uint8_t) * width * height * 3);
                frame->width_alloc = width;
                frame->height_alloc = height;
            }
            frame->width = width;
            frame->height = height;
        }
        mutex.unlock();

        frame->is_sorted = false;
        frame->diff_residual = 0;
        frame->prev_total_residual = 0;
        frame->offset = 0;

        return frame;
    };

    void FreeFrame(EngorgioFrame *frame)
    {
        mutex.lock();
        free_frames.push_back(frame);
        mutex.unlock();
    }

    void FreeFrames(std::vector<EngorgioFrame *> frames)
    {
        mutex.lock();
        for (auto f : frames)
        {
            free_frames.push_back(f);
        }
        mutex.unlock();
    }
};

struct DecodeLatencyLog{
    int video_index, super_index;
    std::chrono::system_clock::time_point start, end;

    DecodeLatencyLog(int video_index_, int super_index_, std::chrono::system_clock::time_point start_, std::chrono::system_clock::time_point end_)
    {
        video_index = video_index_;
        super_index = super_index_;
        start = start_;
        end = end_;
    }
};

struct AnchorIndexLog{
    int video_index, super_index;
    int curr_epoch;

    AnchorIndexLog(int video_index_, int super_index_, int curr_epoch_)
    {
        video_index = video_index_;
        super_index = super_index_;
        curr_epoch = curr_epoch_;
    }
};

struct FrameIndexLog{
    int video_index, super_index;
    int count;

    FrameIndexLog(int video_index_, int super_index_, int curr_epoch_)
    {
        video_index = video_index_;
        super_index = super_index_;
        count = curr_epoch_;
    }
};

    // stream information
struct EngorgioStream {
    int avg_size;
    std::deque<int> sizes;
    std::string content;
    int gop;
    EngorgioFramePool *framepool;
    std::shared_mutex mutex; // mutex to modify frames
    
    // decoder engine
    int worker_index; // decode worker index
    vpx_codec_ctx_t* decoder = nullptr;
    std::deque<EngorgioFrame*> frames;
    
    // anchor engine
    std::vector<EngorgioFrame*> anchors;
    EngorgioModel *model;
    int prev_total_residual = 0;

    // logging
    std::deque<DecodeLatencyLog*> dlatency_logs;
    std::deque<AnchorIndexLog*> aaindex_logs;
    std::deque<FrameIndexLog*> afindex_logs;

    EngorgioStream(int gop_, std::string &content_, EngorgioModel *model_);
    // {
    //     avg_size = 0;
    //     content = content_;
    //     framepool = new EngorgioFramePool(INIT_NUM_FRAMES, 1920, 1080);
    //     decoder = nullptr;
    //     prev_total_residual = 0;
    //     gop = gop_;
    //     model = model_;
    //     anchors.reserve(INIT_NUM_FRAMES);
    // }

    ~EngorgioStream();
    // {
    //     delete framepool;
    //     delete model;

    //     for (auto log : dlatency_logs)
    //         delete log;
    //     for (auto log : aaindex_logs)
    //         delete log;
    //     for (auto log : afindex_logs)
    //         delete log;
    // }
};

// TODO: replace list with deque
struct EngorgioStreamContext {
    std::vector<EngorgioStream*> streams; // TODO: replace vector with list 
    std::deque<int> free_ids; // linked list to allocate stream ids
    std::vector<int> active_ids;
    std::mutex mutex;

    EngorgioStreamContext(int num_streams)
    {
        for (int i = 0; i < num_streams; i++)
        {
            streams.push_back(nullptr);
            free_ids.push_back(i);
        }
    }

    ~EngorgioStreamContext()
    {
        EngorgioStream* stream;
        auto it = active_ids.begin();
        while (it != active_ids.end())
        {
            stream = streams[*it];
            while(!(stream->decoder == nullptr && stream->frames.size() == 0 && stream->anchors.size() == 0))
            {}
        
            delete stream;
            mutex.lock();
            free_ids.push_back(*it);
            it = active_ids.erase(it);
            mutex.unlock();
        }
    }
};

