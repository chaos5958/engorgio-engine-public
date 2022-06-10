#pragma once
#include "control_common.h"
#include <cassert>
#include <vector>
#include <thread>
#include <list>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <filesystem>
#include <string>
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"

namespace fs = std::filesystem;

enum class DecodeType : int
{
	kInit = 0,
	kDecode = 1,
	kDestroy = 2,
    kJoin = 3,
};

struct DecodeEvent
{
    // init, decode, destroy
    DecodeType type;
    int stream_id;

    // decode
    uint8_t* buf;
    int len;
};

struct DecodeWorker 
{
    std::vector<DecodeEvent> events;
    std::shared_mutex mutex;
    int num_streams = 0;
    std::thread *thread;

    // logging
    bool is_first = true;
    int num_frames = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
};

struct DecodeEngineProfile {
    int num_workers;
    bool save_log;
    std::vector<int> thread_indexes;
    std::filesystem::path log_dir; 
    std::chrono::system_clock::time_point start;

    DecodeEngineProfile(int num_workers_, bool save_log_, std::chrono::system_clock::time_point start_, std::vector<int> thread_indexes_ = {}, std::filesystem::path log_dir_ = std::filesystem::path(""))
    {
        assert(save_log_);
        num_workers = num_workers_;
        save_log = save_log_;
        start = start_;
        log_dir = log_dir_;
        
        if (thread_indexes_.size() == 0)
        {
            for (int i = 0; i < num_workers; i++)
            {
                thread_indexes.push_back(i);
            }
        }
        else
        {
            assert((std::size_t) num_workers_ == thread_indexes_.size());
            thread_indexes = thread_indexes_;
        }

    }

    DecodeEngineProfile(int num_workers_, std::vector<int> thread_indexes_ = {})
    {
        num_workers = num_workers_;
        save_log = false;
        if (thread_indexes_.size() == 0)
        {
            for (int i = 0; i < num_workers; i++)
            {
                thread_indexes.push_back(i);
            }
        }
        else
        {
            assert((std::size_t) num_workers_ == thread_indexes_.size());
            thread_indexes = thread_indexes_;
        }    
    }
};



class DecodeEngine {
private:
    // codec setting
    bool disable_loop_filter_ = false;
    bool enable_multi_thread_ = false;
    int num_threads_; // threads per worker

    // engine setting
    int num_workers_; 
    std::vector<DecodeWorker*> workers_;
    std::shared_mutex mutex_;
    EngorgioStreamContext *stream_context_;

    // debugging
    bool save_log_;
    fs::path log_dir_;
    std::chrono::system_clock::time_point start_;

    void DecodeEventHandler(DecodeWorker &worker, int index);
    void DecoderInit(DecodeEvent &event);
    void DecoderDecode(DecodeEvent &event, int index);
    void DecoderDestroy(DecodeEvent &event);

public:
    // DecodeEngine(int num_workers, std::vector<Stream*> &streams);
    DecodeEngine(DecodeEngineProfile &profile, EngorgioStreamContext *stream_context);
    ~DecodeEngine();
    int DecoderInit(int stream_id); // 1. init a decoder, 2. allocate a thread
    int DecoderDestroy(int stream_id);
    int DecoderDecode(int stream_id, uint8_t *buf, int len); // 1. find a thread, 2. push an event to a list 
    void SaveLog(int stream_id);
    void SaveLog();
};
