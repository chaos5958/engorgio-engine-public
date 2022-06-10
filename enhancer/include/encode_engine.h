#pragma once

#include <cassert>
#include <thread>
#include <vector>
#include <turbojpeg.h>
#include <filesystem>
#include "control_common.h"

// #include "kdu_functions.h"

struct JPEGFrame
{
    int stream_id_;
    int curr_video_frame_, curr_super_frame_;
    unsigned char *encoded_buffer_;
    unsigned long jpg_file_len_;

    JPEGFrame()
    {
        encoded_buffer_ = nullptr;
    }

    JPEGFrame(int stream_id, int curr_video_frame, int curr_super_frame,
                unsigned char *encoded_buffer, unsigned long jpg_file_len)
    {
        stream_id_ = stream_id;
        curr_video_frame_ = curr_video_frame;
        curr_super_frame_ = curr_super_frame;
        encoded_buffer_ = nullptr;
        jpg_file_len_ = jpg_file_len;
    }
};

struct JPEGFramePool
{
    int subsampling_;
    std::mutex mutex_;
    std::deque<JPEGFrame*> all_frames_;
    std::deque<JPEGFrame*> free_frames_;

    JPEGFramePool(int subsampling, int num_frames, int width, int height)
    {
        JPEGFrame *frame;
        subsampling_ = subsampling;
        unsigned long size = tjBufSize(width, height, subsampling);
        for (int i = 0; i < num_frames; i++)
        {
            frame = new JPEGFrame();
            frame->encoded_buffer_ = (unsigned char*) malloc(size);
            frame->jpg_file_len_ = size;
            all_frames_.push_back(frame);
            free_frames_.push_back(frame);
        }
    };

    ~JPEGFramePool()
    {
        while(all_frames_.size() != free_frames_.size())
        {    
        }
        // TODO: validate this
        for (auto f : all_frames_)
        {
            if (f)
                delete f;
        }
    };

    int Size()
    {
        return all_frames_.size();
    };

    JPEGFrame* GetFrame(int width, int height)
    {
        JPEGFrame *frame;
        unsigned long size = tjBufSize(width, height, subsampling_);

        mutex_.lock();
        if (free_frames_.size() == 0)
        {
            frame = new JPEGFrame();
            frame->encoded_buffer_ = (unsigned char*) malloc(size);
            frame->jpg_file_len_ = size;
            all_frames_.push_back(frame);
        }
        else
        {
            frame = free_frames_.front();
            free_frames_.pop_front();
            if (size > frame->jpg_file_len_)
            {
                frame->encoded_buffer_ = (unsigned char*) realloc(frame->encoded_buffer_, size);
            }
        }
        mutex_.unlock();
        return frame;
    };

    void FreeFrame(JPEGFrame *frame)
    {
        mutex_.lock();
        free_frames_.push_back(frame);
        mutex_.unlock();
    }
};

enum class JPEGEventType : int
{
	kEncode = 1,
    kSave = 2,
    kJoin = 3,
    // youngmok
    kEncodeKDU = 4,
};


struct JPEGEncodeEvent
{
    JPEGEventType type_;
    int stream_id_;
    std::vector<EngorgioFrame*> frames_;
};

struct JPEGSaveEvent
{
    JPEGEventType type_;
    int stream_id_;
    std::deque<JPEGFrame*> frames_;
};

struct JPEGProfile{
    int num_encode_workers_;
    TJSAMP subsampling_;
    int qp_;
    int flag_;
    bool save_log_, save_image_;
    std::filesystem::path log_dir_;
    std::chrono::system_clock::time_point start_;
    std::vector<int> thread_indexes_;

    JPEGProfile(int qp, int num_workers, std::vector<int> thread_indexes = {})
    {
        num_encode_workers_ = num_workers;
        subsampling_ = TJSAMP_444;
        flag_ = TJFLAG_FASTDCT;
        qp_ = qp;
        save_log_ = false;
        
        if (thread_indexes.size() == 0)
        {
            for (int i = 0; i < num_workers + 1; i++)
            {
                thread_indexes_.push_back(i);
            }
        }
        else
        {
            assert(thread_indexes.size() == (std::size_t) num_workers);
            thread_indexes_ = thread_indexes;
        }
    }

    JPEGProfile(int qp, int num_workers, bool save_log, bool save_image, std::chrono::system_clock::time_point start, std::vector<int> thread_indexes = {}, std::filesystem::path save_dir=std::filesystem::path(""))
    {
        num_encode_workers_ = num_workers;
        subsampling_ = TJSAMP_444;
        flag_ = TJFLAG_FASTDCT;
        qp_ = qp;
        log_dir_ = save_dir;
        save_log_ = save_log;
        save_image_ = save_image;
        start_ = start;

        if (thread_indexes.size() == 0)
        {
            for (int i = 0; i < num_workers + 1; i++)
            {
                thread_indexes_.push_back(i);
            }
        }
        else
        {
            if (save_image)
                assert(thread_indexes.size() == (std::size_t) num_workers + 1);
            else
                assert(thread_indexes.size() == (std::size_t) num_workers);
            thread_indexes_ = thread_indexes;
        }
    }
};

struct JPEGQueryLog{
    int stream_id;
    std::vector<std::chrono::system_clock::time_point> starts, ends;
    std::vector<int> video_indexes;
    std::vector<int> super_indexes;
};

struct JPEGFrameLog{
    int video_index, super_index;
    std::chrono::system_clock::time_point start, end;

    JPEGFrameLog(JPEGQueryLog *query_log, int i)
    {
        video_index = query_log->video_indexes[i];
        super_index = query_log->super_indexes[i];
        start = query_log->starts[i];
        end = query_log->ends[i];
    }
};


class JPEGEncodeEngine{
private:
    int num_encode_workers_;
    int num_save_workers_;
    TJSAMP subsampling_;
    int qp_;
    int flag_;
    JPEGFramePool *framepool_;
    std::vector<int> thread_indexes_;

    // logging
    std::filesystem::path log_dir_;
    std::chrono::system_clock::time_point start_;
    std::vector<std::deque<JPEGQueryLog*>> logs_;

    // debugging
    std::mutex encode_count_mutex_, save_count_mutex_;
    int encode_count_, save_count_;

    std::vector<std::thread*> encode_workers_;
    std::deque<JPEGEncodeEvent> encode_events_;
    std::mutex encode_mutex_;
    std::vector<std::thread*> save_workers_;
    std::deque<JPEGSaveEvent> save_events_;
    std::mutex save_mutex_;

    void EncodeHandler(int index);
    

    void Encode_kakadu(int index, JPEGEncodeEvent &event);

    void Encode(int index, JPEGEncodeEvent &event);
    void SaveImageHandler(int index);
    void SaveImage(JPEGSaveEvent &event);
    void SaveLog();


public:
    bool save_image_, save_log_;

    JPEGEncodeEngine(JPEGProfile &profile);
    ~JPEGEncodeEngine();

    bool EncodeFinished(int num_frames);
    bool SaveFinished(int num_frames);
    void Encode(int stream_id, std::vector<EngorgioFrame*> &frames);
    void Encode_kakadu(int stream_id, std::vector<EngorgioFrame*> &frames);
};
