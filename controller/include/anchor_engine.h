#pragma once 
#include <cassert>
#include <vector> 
#include <thread>
#include <map>
#include <deque>
#include <filesystem>
#include "control_common.h"
#include "neural_enhancer.h"

enum class SelectPolicy : int
{
    ENGORGIO_BUDGET,
    ENGORGIO_FRACTION,
    PERFRAME,
    SELECTIVE
};

enum class SortType : int
{
	kSort,
    kJoin
};

struct SortEvent
{
    SortType type;
    int stream_id;
};

struct CompareFrame{
    bool operator()(EngorgioFrame *f1, EngorgioFrame *f2){
        return f1->diff_residual < f2->diff_residual;
    };
};

struct AnchorEngineProfile {
    SelectPolicy policy;
    bool save_log;
    int gop; 
    std::filesystem::path log_dir; 
    std::chrono::system_clock::time_point start;
    int thread_index;
    double fraction;

    AnchorEngineProfile(SelectPolicy policy_, bool save_log_, std::chrono::system_clock::time_point start_, int gop_, int thread_index_ = 0, std::filesystem::path log_dir_ = std::filesystem::path(""))
    {
        assert(save_log_);
        policy = policy_;
        save_log = save_log_;
        start = start_;
        gop = gop_;
        log_dir = log_dir_;
        thread_index = thread_index_;
        fraction = 0;
    }

    AnchorEngineProfile(SelectPolicy policy_, int gop_, int thread_index_ = 0, std::filesystem::path log_dir_ = std::filesystem::path(""))
    {
        policy = policy_;
        save_log = false;
        gop = gop_;
        log_dir = log_dir_;
        thread_index = thread_index_;
        fraction = 0;
    }

    AnchorEngineProfile(SelectPolicy policy_, bool save_log_, std::chrono::system_clock::time_point start_, int gop_, double fraction_, int thread_index_ = 0, std::filesystem::path log_dir_ = std::filesystem::path(""))
    {
        // assert(save_log_ && policy_ == SelectPolicy::ENGORGIO_FRACTION);
        // assert(fraction_ >= 0 && fraction_ < 1);
        policy = policy_;
        save_log = save_log_;
        start = start_;
        gop = gop_;
        log_dir = log_dir_;
        thread_index = thread_index_;
        fraction = fraction_;
    }

    AnchorEngineProfile(SelectPolicy policy_, int gop_, double fraction_, int thread_index_ = 0, std::filesystem::path log_dir_ = std::filesystem::path(""))
    {
        assert(policy_ == SelectPolicy::ENGORGIO_FRACTION);
        assert(fraction_ > 0 && fraction_ < 1);
        policy = policy_;
        save_log = false;
        gop = gop_;
        log_dir = log_dir_;
        thread_index = thread_index_;
        fraction = fraction_;
    }
};

struct AnchorLog{
    std::map<int, std::deque<std::string>> anchor_indexes;
    double sort_latency; // ms
    double select_latency; // ms
    double free_latency;
};

struct AnchorScheduleLog{
    double total_budget, used_budget;
    int num_anchors, count;
    std::chrono::system_clock::time_point start, end;
    
    AnchorScheduleLog(double total_budget_, double used_budget_, int count_, int num_anchors_, 
                        std::chrono::system_clock::time_point start_, std::chrono::system_clock::time_point end_)
    {
        total_budget = total_budget_;
        used_budget = used_budget_;
        num_anchors = num_anchors_;
        count = count_;
        start = start_;
        end  = end_;
    }
};

struct EngorgioContext
{
    // Per-stream 
    std::vector<EngorgioFrame*> target_frames, key_frames, altref_frames, normal_frames;
    std::vector<int> total_residuals;

    // Global
    std::deque<EngorgioFrame*> global_key_frames; 
    std::priority_queue<EngorgioFrame*, std::vector<EngorgioFrame*>, CompareFrame> global_altref_frames, global_normal_frames;

    EngorgioContext(int gop)
    {
        int init_frames = gop * 1.5;
        target_frames.reserve(init_frames);
        key_frames.reserve(init_frames);
        altref_frames.reserve(init_frames);
        normal_frames.reserve(init_frames);    
        total_residuals.reserve(init_frames);
    }
};

struct SelectiveContext
{
    int num_anchors;
};

class AnchorEngine
{
private:
    // common variables
    SelectPolicy policy_;
    int gop_;
    EngorgioStreamContext *stream_context_;
    std::thread *periodic_thread_;
    bool periodic_;
    int curr_epoch_;
    int thread_index_;
    double fraction_;
    int num_total_anchors_;

    // logging
    bool save_log_;
    std::filesystem::path log_dir_;
    std::deque<AnchorScheduleLog*> schedule_logs_;
    std::chrono::system_clock::time_point start_;

    // engorgio
    EngorgioContext *engorgio_context_;  

    // perframe

    // selective
    SelectiveContext *selective_context_;

    // offline
    NeuralEnhancer *neural_enhancer_;
    std::map<std::pair<int, std::string>, double> dnn_latencies_;



    // engorgio functions
    void RunEngorgio();
    void SortStream(int stream_id);
    void SortGroup(EngorgioStream *stream, std::vector<EngorgioFrame*> &frames, std::vector<EngorgioFrame*> &target_frames, std::vector<int> &total_residuals, int num_anchors, int gop);
    void SaveEngorgioLog();
    void SaveEngorgioLog(int stream_id);

    // per-frame functions
    void RunPerFrame();

    // selective functions
    void RunSelective();
public:
    // common functions
    AnchorEngine(AnchorEngineProfile &profile, EngorgioStreamContext *stream_context);
    ~AnchorEngine();

    void Run();
    void RunPeriodic();

    void SaveLog();
    void SaveLog(int stream_id);

    // offline
    void LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer);
    void LoadDNNLatency(int resolution, std::string &name, double latency);

    // common
    void LaunchPeriodic();
    void StopPeriodic();
    int GetTotalAnchors();
};

