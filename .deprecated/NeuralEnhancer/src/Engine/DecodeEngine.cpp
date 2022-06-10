#include "DecodeEngine.h"

#include <iostream>
#include <limits>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;


using namespace std;

// constructor
DecodeEngine::DecodeEngine(EngineType etype, int threads)
{
    etype_ = etype;
    curr_thread_idx_ = 0;
    num_threads_ = threads;

    // TODO: provide an API to set this
    disable_loop_filter_ = false;
    enable_multi_thread_ = true;
    codec_threads_ = 1;
}

DecodeEngine::DecodeEngine(EngineType etype, int threads, string log_dir): DecodeEngine(etype, threads)
{
    log_dir_ = log_dir;
    if (!fs::exists(log_dir_))
    {
        fs::create_directories(log_dir_);
    }
}


// destructor
DecodeEngine::~DecodeEngine()
{
    Destroy();
}

int DecodeEngine::Build()
{
    // per-stream
    thread_indexes_ = std::vector<int>(MAX_STREAM_ID, -1);
    contexts_ = std::vector<DecodeContext*>(MAX_STREAM_ID, nullptr);
#ifdef LOG_DECODE_ENGINE
    logs_ = std::vector<DecodeLog>(MAX_STREAM_ID);
    start_ = std::chrono::high_resolution_clock::now();
#endif
    std::vector<std::shared_mutex> tmp_smutexes = std::vector<std::shared_mutex>(MAX_STREAM_ID);
    smutexes_.swap(tmp_smutexes);
    
    // per_thread 
    max_threads_ =  std::thread::hardware_concurrency();
    for (int i = 0; i < max_threads_; i++)
    {
        free_thread_indexes_.push_back(i);
    }
    events_ = std::vector<std::vector<DecodeEvent>>(max_threads_, std::vector<DecodeEvent>());
    latencies_ = std::vector<int>(max_threads_, std::numeric_limits<int>::max());
    std::vector<std::thread> tmp_threads = std::vector<std::thread>(max_threads_);
    std::vector<std::shared_mutex> tmp_dmutexes = std::vector<std::shared_mutex>(max_threads_);
    threads_.swap(tmp_threads);
    dmutexes_.swap(tmp_dmutexes);
    
    // create threads
    int idx;
    for (int i = 0; i < num_threads_; i ++) 
    {
        idx = free_thread_indexes_.back();
        free_thread_indexes_.pop_back();
        alloc_thread_indexes_.push_back(idx);
        threads_[idx] = std::thread([this, idx](){this->DecodeEventHandler(idx);}); // TODO: pass frame indexes
    }
    return 0;
}

void DecodeEngine::Destroy()
{
    DecodeEvent event;
    event.dtype = DecodeType::kJoin;
    for (auto idx : alloc_thread_indexes_)
    {
        dmutexes_[idx].lock();
        events_[idx].push_back(event);
        dmutexes_[idx].unlock();
    }

    for (auto idx : alloc_thread_indexes_)
    {
        threads_[idx].join();
    }
}

// TODO: setting
void DecodeEngine::DecoderInit(DecodeEvent &devent)
{
// #ifdef LOG_DECODE_ENGINE
//     auto start = std::chrono::high_resolution_clock::now();
// #endif
    int stream_id = devent.stream_id;

    DecodeContext *context = new DecodeContext();
    context->decoder = new vpx_codec_ctx_t();
    vpx_codec_dec_cfg cfg = {0, 0, 0};
    cfg.threads = codec_threads_;

    vpx_codec_err_t err = vpx_codec_dec_init(context->decoder, &vpx_codec_vp9_dx_algo, &cfg, 0);
    if (err)
    {
        cerr << "Failed to initialize libvpx decoder, error =" << err << endl;
        return;
    }

#ifdef VPX_CTRL_VP9_DECODE_SET_ROW_MT
    err = vpx_codec_control(context->decoder, VP9D_SET_ROW_MT, enable_multi_thread_);
    if (err)
    {
        cerr << "Failed to enable row multi thread mode, error = " << err << endl;
    }
#endif

    if (disable_loop_filter_)
    {
        err = vpx_codec_control(context->decoder, VP9_SET_SKIP_LOOP_FILTER, true);
        if (err)
        {
            cerr << "Failed to shut off libvpx loop filter, error = " << err << endl;
        }
    }
#ifdef VPX_CTRL_VP9_SET_LOOP_FILTER_OPT
    else
    {
        err = vpx_codec_control(context->decoder, VP9D_SET_LOOP_FILTER_OPT, true);
        if (err)
        {
            cerr << "Failed to enable loop filter optimization, error = " << err << endl;
        }
#endif
    }

    // TODO: enable own frame buffer manager
    // err = vpx_codec_set_frame_buffer_functions(
    //     context->decoder, vpx_get_frame_buffer, vpx_release_frame_buffer,
    //     context->buffer_manager);
    // if (err)
    // {
    //     LOGE("Failed to set libvpx frame buffer functions, error = %d.", err);
    // }
    
    // critical section 1: modify context_map
    smutexes_[stream_id].lock();
    contexts_[stream_id] = context;
    smutexes_[stream_id].unlock();
#ifdef LOG_DECODE_ENGINE
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    logs_[stream_id].latencies.clear();
    logs_[stream_id].is_first = true;
    logs_[stream_id].frames = 0;
    // std::cout << "Elapsed time (Init): " << elapsed.count() * 1000 << " ms" << endl;;
#endif
}

void DecodeEngine::DecoderDecode(DecodeEvent &devent)
{
#ifdef LOG_DECODE_ENGINE
    auto start = std::chrono::high_resolution_clock::now();
#endif

    int stream_id = devent.stream_id;
    uint8_t *buf = devent.buf;
    int len = devent.len;

    smutexes_[stream_id].lock_shared();
    DecodeContext *context = contexts_[stream_id];
    smutexes_[stream_id].unlock_shared();
    const vpx_codec_err_t status = vpx_codec_decode(context->decoder, buf, len, NULL, 0);
    if (status != VPX_CODEC_OK)
    {
        cerr << "vpx_codec_decode() failed, status = " << status << endl;
    }

#ifdef LOG_DECODE_ENGINE
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (logs_[stream_id].is_first)
    {
        logs_[stream_id].start = start;
        logs_[stream_id].is_first = false;
    }
    logs_[stream_id].end = end;
    logs_[stream_id].frames += 1;
    logs_[stream_id].latencies.push_back(elapsed.count() * 1000);

    // std::cout << "Elapsed time (Decode): " << elapsed.count() * 1000 << " ms\n";
#endif 
}

void DecodeEngine::DecoderDestroy(DecodeEvent &devent)
{
// #ifdef LOG_DECODE_ENGINE
//     auto start = std::chrono::high_resolution_clock::now();
// #endif

    // TODO: release all frames in memory
    int stream_id = devent.stream_id;
    smutexes_[stream_id].lock();
    DecodeContext *context = contexts_[stream_id];
    contexts_[stream_id] = nullptr;
    smutexes_[stream_id].unlock();

    vpx_codec_destroy(context->decoder);
    delete context;

#ifdef LOG_DECODE_ENGINE
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time (Destroy): " << elapsed.count() * 1000 << " ms\n";

    string log_file = log_dir_ + "/" + "decode_" + to_string(stream_id) + ".txt";
    std::chrono::duration<double> elapsed = logs_[stream_id].end - logs_[stream_id].start;
    double throughput = logs_[stream_id].frames / elapsed.count();
    double avg_latency, sum = 0;
    for (int i = 0; i < logs_[stream_id].latencies.size(); i++)
    {
        sum += logs_[stream_id].latencies[i];
    }
    avg_latency = sum / logs_[stream_id].latencies.size();

    ofstream write_file(log_file.data());
    if (write_file.is_open())
    {
        write_file << "duration\t" << elapsed.count() * 1000 << endl;
        write_file << "frames\t" << logs_[stream_id].frames << endl;
        write_file << "throughput\t" << throughput << endl;
        write_file << "avg latency\t" << avg_latency << endl;
        for (auto latency : logs_[stream_id].latencies)
        {
            write_file << latency << endl;
        }
    }
#endif
}


// TODO: check tasks (other than decode) take less than 0.1ms
void DecodeEngine::DecodeEventHandler(int idx)
{
    DecodeEvent target_event;
    bool handle_event;

    while (1)
    {
        handle_event = false;
        dmutexes_[idx].lock();
        if (!events_[idx].empty())
        {
            target_event = events_[idx].front();
            events_[idx].erase(events_[idx].begin());
            handle_event = true;
        }
        dmutexes_[idx].unlock();

        if (handle_event)
        {
            switch (target_event.dtype)
            {
            case DecodeType::kInit:
                DecoderInit(target_event);
                break;
            case DecodeType::kDecode:
                DecoderDecode(target_event);
                break;
            case DecodeType::kDestroy:
                DecoderDestroy(target_event);
                break;
            case DecodeType::kJoin:
                return;
            default:
                cerr << "Unsupported event type" << endl;
                break;
            }
        }
    }
}


// TODO: mutex
// TODO: measure initialization latency (1. critical: do it on a seperate thread, 2. fast: do it on a decode thread)
// TODO: increase latencies (lock)
int DecodeEngine::DecoderInit(int stream_id)
{
    // select a thread
    int idx;

    gmutex_.lock();
    switch (etype_)
    {
    case EngineType::kBaseline:
        idx = alloc_thread_indexes_[curr_thread_idx_];
        curr_thread_idx_ = (curr_thread_idx_ + 1) % num_threads_;    
        break;
    case EngineType::kOur:
        cerr << "Not implemented yet" << endl;
        // TODO: select a thread with the minimum latency
        // TODO: if min latency + curr latency > threshold - create a new thread
        // TODO: update latency
        return -1;
        break;
    default:
        cerr << "Invalid EngineType" << endl;
        return -1;
    }
    gmutex_.unlock();
    smutexes_[stream_id].lock();
    thread_indexes_[stream_id] = idx;
    smutexes_[stream_id].unlock();

        
    // 2. push an event to a thread
    // TODO: do we need a lockless queue here? 
    DecodeEvent event;
    event.dtype = DecodeType::kInit;
    event.stream_id = stream_id;
    dmutexes_[idx].lock();
    events_[idx].push_back(event);
    dmutexes_[idx].unlock();
    return 0;
}

// TODO: rebalance로 frame이 뒤로 갈 경우, 이를 기억해야 destroy 할 때 도움이 됨
// TODO: decrease latencies (lock)
int DecodeEngine::DecoderDestroy(int stream_id)
{
    // find a thread
    int idx;
    smutexes_[stream_id].lock_shared();
    idx = thread_indexes_[stream_id];
    smutexes_[stream_id].unlock_shared();
    
    // push an event to a thread
    // TODO: do we need a lockless queue here? 
    DecodeEvent event;
    event.dtype = DecodeType::kDestroy;
    event.stream_id = stream_id;
    dmutexes_[idx].lock();
    events_[idx].push_back(event);
    dmutexes_[idx].unlock();

    //TODO: reduce latency (using gmutex_)

    return 0;
}

// TODO: use lock_shared(), unlock_shared()
int DecodeEngine::DecoderDecode(int stream_id, uint8_t *buf, int len)
{
    // find a thread
    int idx;
    smutexes_[stream_id].lock_shared();
    idx = thread_indexes_[stream_id];
    smutexes_[stream_id].unlock_shared();
    
    // push an event to a thread
    // TODO: do we need a lockless queue here? 
    DecodeEvent event;
    event.dtype = DecodeType::kDecode;
    event.stream_id = stream_id;
    event.buf = buf;
    event.len = len;
    dmutexes_[idx].lock();
    events_[idx].push_back(event);
    dmutexes_[idx].unlock();

    return 0;
}

int DecodeEngine::AnchorProcess(int stream_id, int frame_idx, int super_idx)
{
    // 1. call a callback
    // 2. release decoded frames
}

// TODO: design: thread_configure하는 thread를 만들지 1초마다 loop하면서?
int ThreadRebalance()
{
    // 1: calculate diff. btw the estimated and the current throughput
    // 2. create or delete a thread - delete needs to push a event
    // Caution: use a lock only when updating indexes

    // 일정시간 기다렸다가 input 안들어오면 끄도록 업데이트 - 이를 internal에서 구현해야함 kjoin 
}
