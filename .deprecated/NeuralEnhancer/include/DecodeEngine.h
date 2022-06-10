#pragma once

#include <vector>
#include <thread>
#include <list>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <string>
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"

#define LOG_DECODE_ENGINE

// info: https://stackoverflow.com/questions/13955634/thread-safety-of-writing-a-stdvector-vs-plain-array
// TODO: use condition variables (for infrequent updates)

// gmutex_: mutex for modifying latency infromation 
// smutex_: mutexes for accessing/modifying per-stream information
// dmutex_: mutexes for accessing/modifying per-thread information

const int MAX_STREAM_ID = 10000; // TODO: move it to the sr engine header 

enum class DecodeType : int
{
	kInit = 0,
	kDecode = 1,
	kDestroy = 2,
    kJoin = 3,
};

enum class EngineType : int
{
	kBaseline = 0,
	kOur = 1,
};


struct DecodeEvent
{
    // init, decode, destroy
    DecodeType dtype;
    int stream_id;

    // decode
    uint8_t* buf;


    int len;
};

struct DecodeContext{
    // TODO: JniContext + Decoded frames + Per-stream lock
    DecodeContext() {}
    ~DecodeContext() {}

    vpx_codec_ctx_t* decoder = NULL;
};

struct DecodeLog {
    std::chrono::system_clock::time_point start, end;
    int frames;
    std::vector<double> latencies;
    bool is_first;
};

// TODO: codec information (?) which to hold (?)
// TODO: use condition variable 

class DecodeEngine {
private:
    // codec setting
    bool disable_loop_filter_;
    bool enable_multi_thread_;
    int codec_threads_;

    // engine setting
    EngineType etype_;
    int num_threads_, max_threads_; // max_threads = num cores
    int curr_thread_idx_; // used for round-robin  
    std::shared_mutex gmutex_;

    // log
    std::vector<DecodeLog> logs_;
    std::string log_dir_;
    std::chrono::system_clock::time_point start_;
    
    // per-stream
    std::vector<int> thread_indexes_;
    std::vector<DecodeContext*> contexts_;
    std::vector<std::shared_mutex> smutexes_;

    // per-thread
    std::vector<int> free_thread_indexes_;
    std::vector<int> alloc_thread_indexes_;
    std::vector<std::vector<DecodeEvent>> events_;
    std::vector<std::shared_mutex> dmutexes_;
    std::vector<std::thread> threads_;
    std::vector<int> latencies_;  // default - inf 

    void DecodeEventHandler(int thread_idx);
    void AnchorEventHandler(int thread_idx);
    void DecoderInit(DecodeEvent &devent);
    void DecoderDecode(DecodeEvent &devent);
    void DecoderDestroy(DecodeEvent &devent);

public:
    DecodeEngine(EngineType etype, int threads);
    DecodeEngine(EngineType etype, int threads, std::string log_dir);
    ~DecodeEngine();
    int Build(); 
    void Destroy();
    int DecoderInit(int stream_id); // 1. init a decoder, 2. allocate a thread
    int DecoderDestroy(int stream_id);
    int DecoderDecode(int stream_id, uint8_t *buf, int len); // 1. find a thread, 2. push an event to a list 
    int AnchorProcess(int stream_id, int frame_idx, int super_idx);
    int ThreadRebalance();
};
