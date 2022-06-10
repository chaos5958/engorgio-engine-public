#include "anchor_engine.h"
#include "date/date.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <thread>
#include "neural_enhancer.h"
#include "enhancer_common.h"

namespace fs = std::filesystem;

using namespace std;
using namespace date;
using namespace std::chrono;

// TODO: remove
const int NUM_ANCHORS = 4;
const double MAX_SORT_RATIO = 0.2;
// const int INTERVAL_IN_MS = 1000;
const double INTERVAL_IN_MS = 666.6666;
const int INTERVAL_IN_MICROS = int(666.6666* 1000);
const int INTERVAL_PER_FRAME_IN_MiCROS = 200 * 1000;
const int INTERVAL_PER_FRAME_IN_MS = 200;
const int FRAMES_PER_SECOND = 60;

AnchorEngine::AnchorEngine(AnchorEngineProfile &profile, EngorgioStreamContext *stream_context)
{
    policy_ = profile.policy;
    gop_ = profile.gop;
    save_log_ = profile.save_log;
    log_dir_ = profile.log_dir;
    start_ = profile.start;
    stream_context_ = stream_context;
    periodic_ = false;
    curr_epoch_ = 0;
    thread_index_ = profile.thread_index;
    fraction_ = profile.fraction;

    num_total_anchors_ = 0;
    engorgio_context_ = nullptr;
    selective_context_ = nullptr;
    neural_enhancer_ = nullptr;
    switch(policy_)
    {
        case SelectPolicy::ENGORGIO_BUDGET:
        case SelectPolicy::ENGORGIO_FRACTION:
            engorgio_context_ = new EngorgioContext(gop_);
            break;
        case SelectPolicy::SELECTIVE:

        case SelectPolicy::PERFRAME:
            break;
        default:
            throw std::invalid_argument("Invalid policy");
    }
}

AnchorEngine::~AnchorEngine()
{
    if (periodic_)
        StopPeriodic();

    SaveLog();
    for (auto log : schedule_logs_)
    {
        delete log;
    }

    switch(policy_)
    {
        case SelectPolicy::ENGORGIO_BUDGET:
        case SelectPolicy::ENGORGIO_FRACTION:
            delete engorgio_context_;
            break;
        case SelectPolicy::SELECTIVE:
        case SelectPolicy::PERFRAME:
            break;
        // default:
        //     throw std::invalid_argument("Invalid policy");
    }
}

void AnchorEngine::SortGroup(EngorgioStream *stream, std::vector<EngorgioFrame*> &frames, std::vector<EngorgioFrame*> &target_frames, std::vector<int> &total_residuals, int num_frames, int gop)
{
    int curr_index, next_index;
    int max_curr_index, max_next_index;
    double diff_residual, max_diff_residual;
    bool exists;

    auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "bitrate: " << stream->avg_size * 60 * 8 / 1024 << "," << std::endl;

    // select a frame
    while (num_frames > 0)
    {
        max_diff_residual = -1;
        for (size_t i = 0; i < target_frames.size(); i++)
        {
            if (!target_frames[i]->is_sorted)
            {
                exists = false;
                curr_index = target_frames[i]->offset;
                for (size_t j = curr_index + 1; j < frames.size(); j++)
                {
                    if (frames[j]->is_sorted or frames[j]->frame_type == 0)
                    {
                        exists = true;
                        next_index = j;
                        break;
                    }
                }

                if (!exists)
                {
                    if (gop > 0) 
                        next_index = curr_index + (gop - target_frames[i]->current_video_frame % gop);
                    else 
                        next_index = frames.size();
                }

                // design: normalize by bitrate
                diff_residual = (next_index - curr_index) * total_residuals[curr_index] / stream->avg_size;

                if (diff_residual > max_diff_residual) 
                {
                    max_diff_residual = diff_residual;
                    max_curr_index = curr_index;
                    max_next_index = std::min(next_index, (int) frames.size());
                }
            }
        }

        // update residual, add it to a sorted
        int diff_residual = total_residuals[max_curr_index];
        for (int i = max_curr_index; i < max_next_index; i++)
            total_residuals[i] -= diff_residual;
        int prev_total_residual = total_residuals.back();

        EngorgioFrame *sorted_frame = frames[max_curr_index];
        sorted_frame->is_sorted = true;
        // TODO: normalize by bitrate not size
        sorted_frame->diff_residual = max_diff_residual;
        sorted_frame->prev_total_residual = prev_total_residual;

        num_frames -= 1;

        // cout << sorted_frame->current_video_frame << "," << sorted_frame->current_super_frame << ',' << sorted_frame->diff_residual << endl;
        // cout << sorted_frame->current_super_frame << endl;
        // for (auto residual : total_residuals)
        // {
        //     cout << residual << ',';
        // }
        // cout << endl;
        // for (auto r : total_residuals)
        // {
        //     std::cout << std::to_string(r) << ',' ;
        // }
        // std::cout << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (SortFrames): " << elapsed.count() * 1000 << "ms" << std::endl;
}

void AnchorEngine::SortStream(int stream_id)
{
    EngorgioStream *stream = stream_context_->streams[stream_id];
    std::vector<EngorgioFrame*> &target_frames = engorgio_context_->target_frames;
    std::vector<EngorgioFrame*> &key_frames = engorgio_context_->key_frames ; 
    std::vector<EngorgioFrame*> &altref_frames = engorgio_context_->altref_frames; 
    std::vector<EngorgioFrame*> &normal_frames = engorgio_context_->normal_frames; 
    std::deque<EngorgioFrame*> &global_key_frames = engorgio_context_->global_key_frames;
    std::vector<int> &total_residuals = engorgio_context_->total_residuals;
    std::priority_queue<EngorgioFrame*, std::vector<EngorgioFrame*>, CompareFrame> &global_altref_frames = engorgio_context_->global_altref_frames;
    std::priority_queue<EngorgioFrame*, std::vector<EngorgioFrame*>, CompareFrame> &global_normal_frames = engorgio_context_->global_normal_frames;
    
    // step1: setup
    auto start1 = std::chrono::high_resolution_clock::now(); 
    EngorgioFrame *frame;
    stream->mutex.lock();
    int num_frames = stream->frames.size();
    int total_residual = stream->prev_total_residual;
    for (int i = 0; i < num_frames; i++)
    {
        frame = stream->frames.front();
        frame->offset = i;
        target_frames.push_back(frame);
        stream->frames.pop_front();
        
        switch (frame->frame_type)
        {
        case 0:
            key_frames.push_back(frame);
            break;
        case 1:
            altref_frames.push_back(frame);
            break;
        case 2:
            normal_frames.push_back(frame);
            break;
        default:
            break;
        } 

        if (frame->frame_type != 0)
            total_residual += frame->residual_size;
        else
            total_residual = 0;
        total_residuals.push_back(total_residual);
    }
    stream->mutex.unlock();

    // std::cout << "Sort stream: " << stream_id << "," << target_frames.size() 
    //             << "," << key_frames.size() << "," << altref_frames.size() << "," << normal_frames.size() << std::endl;
    

    // auto startx = std::chrono::high_resolution_clock::now(); 
    FrameIndexLog *log;
    if (save_log_)
    {
        for (auto frame : target_frames)
        {
            log = new FrameIndexLog(frame->current_video_frame, frame->current_super_frame, curr_epoch_);
            stream->afindex_logs.push_back(log);
        }
    }
    // auto endx = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsedx = endx - startx;
    // std::cout << "Latency (Logging): " << elapsedx.count() * 1000 << "ms" << std::endl;
    
    // sort and add frames: altref frames, normal frames
    int num_target_frames = int(round(num_frames * MAX_SORT_RATIO));
    int num_altref_frames = int(std::min(num_target_frames, (int) altref_frames.size()));
    int num_normal_frames = int(std::min(num_target_frames - num_altref_frames, (int) normal_frames.size()));

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    // std::cout << "Latency (Sort, Step1): " << elapsed.count() * 1000 << "ms" << std::endl;

    // std::cout << "num_frames: " << num_frames << std::endl;
    // std::cout << "num_target_frames: " << num_target_frames << std::endl;
    // std::cout << "num_altref_frames: " << num_altref_frames << std::endl;
    // std::cout << "num_normal_frames: " << num_normal_frames << std::endl;

    // step 2: sort per group
    auto start2 = std::chrono::high_resolution_clock::now();
    SortGroup(stream, target_frames, altref_frames, total_residuals, num_altref_frames, stream->gop);
    SortGroup(stream, target_frames, normal_frames, total_residuals, num_normal_frames, stream->gop);
    target_frames.clear();
    total_residuals.clear();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = end2 - start2;
    // std::cout << "Latency (Sort, Step2): " << elapsed.count() * 1000 << "ms" << std::endl;

    // step 3: add frames to global queues
    auto start3 = std::chrono::high_resolution_clock::now();
    size_t size = key_frames.size();
    // std::cout << size << std::endl;
    while (size > 0)
    {
        frame = key_frames.back();
        key_frames.pop_back();
        global_key_frames.push_back(frame);
        size--;
    }
    
    size = altref_frames.size();
    // std::cout << size << std::endl;
    while (size > 0)
    {
        frame = altref_frames.back();
        altref_frames.pop_back();
        if (frame->is_sorted)
            global_altref_frames.push(frame);
        else
            stream->framepool->FreeFrame(frame);
        size--;
    }
    size = normal_frames.size();
    // std::cout << size << std::endl;
    while (size > 0)
    {
        frame = normal_frames.back();
        normal_frames.pop_back();
        if (frame->is_sorted)
            global_normal_frames.push(frame);
        else
            stream->framepool->FreeFrame(frame);
        size--;
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto elapsed3 = end3 - start3;
    // std::cout << "Latency (Sort, Step3): " << elapsed.count() * 1000 << "ms" << std::endl;
}

void AnchorEngine::RunEngorgio()
{ 
    std::chrono::system_clock::time_point schedule_start = std::chrono::system_clock::now();
    std::vector<EngorgioStream*> streams = stream_context_->streams;
    std::mutex &smutex = stream_context_->mutex;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::vector<int> target_ids;

    // step1: sort
    auto start1 = std::chrono::high_resolution_clock::now();   
    smutex.lock();
    for (auto id : active_ids)
    {
        if (streams[id]->frames.size() > 0)
            target_ids.push_back(id);
    }
    smutex.unlock();
    for (auto id: target_ids)
    {
        // std::cout << id << "," << streams[id]->anchors.size() << std::endl;
        SortStream(id);
        assert(streams[id]->anchors.size() == 0); // Validate
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    // std::cout << "Sort: " << elapsed1.count() * 1000 << "ms" << std::endl;
    // std::cout << engorgio_context_->key_frames.size() << std::endl;
    // std::cout << engorgio_context_->altref_frames.size() << std::endl;
    // std::cout << engorgio_context_->normal_frames.size() << std::endl;

    // step2: select
    auto start2 = std::chrono::high_resolution_clock::now();   
    EngorgioFrame *frame;
    EngorgioStream *stream;
    EngorgioModel *model;

    // Validate 1 (pass)
    // std::cout << neural_enhancer_->GetGPUs() << std::endl;
    // frame = engorgio_context_->global_key_frames.back();
    // stream = streams[frame->stream_id];
    // std::cout << streams[0]->model->name_ << std::endl;
    // std::cout << dnn_latencies_[std::make_pair(frame->height, streams[0]->model->name_)] << std::endl;
    // while (1)
    // {};
    // std::cout << "key: " << engorgio_context_->global_key_frames.size() << std::endl;
    // std::cout << "altref: " << engorgio_context_->global_altref_frames.size() << std::endl;
    // std::cout << "normal2: " << engorgio_context_->global_normal_frames.size() << std::endl;

    double budget_in_ms, budget_total;
    int num_total_anchors = 0;
    if (neural_enhancer_)
    {
        // select anchor frames
        assert(policy_ == SelectPolicy::ENGORGIO_BUDGET || policy_ == SelectPolicy::ENGORGIO_FRACTION);
        if (policy_ == SelectPolicy::ENGORGIO_BUDGET)
        {
            budget_in_ms = neural_enhancer_->GetGPUs() * INTERVAL_IN_MS;
            budget_total = budget_in_ms;
            bool exit = false;
            num_total_anchors = 0;
            while (!exit)
            {
                if (engorgio_context_->global_key_frames.size() > 0) 
                {
                    frame = engorgio_context_->global_key_frames.back();
                    engorgio_context_->global_key_frames.pop_back();
                }
                else if (engorgio_context_->global_altref_frames.size() > 0)
                {
                    frame = engorgio_context_->global_altref_frames.top();
                    engorgio_context_->global_altref_frames.pop();
                }
                else if (engorgio_context_->global_normal_frames.size() > 0)
                {
                    frame = engorgio_context_->global_normal_frames.top();
                    engorgio_context_->global_normal_frames.pop();
                }
                else
                {
                    std::cout << "Not enought frame" << std::endl;
                    exit = true;
                    continue;
                }
                
                stream = streams[frame->stream_id];
                model = stream->model;
                double lantecy_in_ms = dnn_latencies_[std::make_pair(frame->height, model->name_)];
                if (budget_in_ms > lantecy_in_ms)
                {
                    stream->anchors.push_back(frame);
                    stream->prev_total_residual = frame->prev_total_residual;
                    num_total_anchors += 1;
                    budget_in_ms -= lantecy_in_ms;
                }
                else
                {
                    stream->framepool->FreeFrame(frame);
                    exit = true;
                }
            }
            num_total_anchors_ += num_total_anchors;
        }
        else if (policy_ == SelectPolicy::ENGORGIO_FRACTION)
        {
            // TODO: Start from here
            num_total_anchors = round(target_ids.size() *  (INTERVAL_IN_MS / 1000) * FRAMES_PER_SECOND * fraction_);
            num_total_anchors_ += num_total_anchors;
            // std::cout << "num_total_anchors: " << num_total_anchors << "," << (INTERVAL_IN_MS / 1000) * FRAMES_PER_SECOND * fraction_ << std::endl;
            frame = nullptr;
            while (num_total_anchors)
            {
                if (engorgio_context_->global_key_frames.size() > 0) 
                {
                    frame = engorgio_context_->global_key_frames.back();
                    engorgio_context_->global_key_frames.pop_back();

                }
                else if (engorgio_context_->global_altref_frames.size() > 0)
                {
                    frame = engorgio_context_->global_altref_frames.top();
                    engorgio_context_->global_altref_frames.pop();
                }
                else if (engorgio_context_->global_normal_frames.size() > 0)
                {
                    frame = engorgio_context_->global_normal_frames.top();
                    engorgio_context_->global_normal_frames.pop();
                }
                if (frame != nullptr)
                {
                    stream = streams[frame->stream_id];
                    stream->anchors.push_back(frame);
                    stream->prev_total_residual = frame->prev_total_residual;
                }
                num_total_anchors--;
            }
        }

        // validate 2 (Use full budget)
        // std::cout << budget_in_ms << std::endl;
        // std::cout << num_anchors << std::endl;
        // while (1)
        // {}
        
        // validate 3 (Query)
        // Print quereis at the neural enhancer 
        // TODO: rollback
        // auto deadline =  std::chrono::system_clock::now() + std::chrono::milliseconds(INTERVAL_IN_MS);
        auto deadline =  std::chrono::system_clock::now() + std::chrono::microseconds(INTERVAL_IN_MICROS);
        
        for (auto id: target_ids)
        {
            // std::cout << "Query sents" << std::endl;
            stream = streams[id];
            if (stream->anchors.size() > 0)
                neural_enhancer_->Process(id, stream->model, stream->framepool, stream->anchors, deadline, false);
            // stream->anchors.clear(); // TODO: move this after logging
        }
    }
    else
    {
        num_total_anchors = target_ids.size() * NUM_ANCHORS;
        num_total_anchors_ += num_total_anchors;
        while (num_total_anchors)
        {
            if (engorgio_context_->global_key_frames.size() > 0) 
            {
                frame = engorgio_context_->global_key_frames.back();
                engorgio_context_->global_key_frames.pop_back();

            }
            else if (engorgio_context_->global_altref_frames.size() > 0)
            {
                frame = engorgio_context_->global_altref_frames.top();
                engorgio_context_->global_altref_frames.pop();
            }
            else 
            {
                frame = engorgio_context_->global_normal_frames.top();
                engorgio_context_->global_normal_frames.pop();
            }
            stream = streams[frame->stream_id];
            stream->anchors.push_back(frame);
            stream->prev_total_residual = frame->prev_total_residual;
            num_total_anchors--;
        }
    }
    std::chrono::system_clock::time_point schedule_end = std::chrono::system_clock::now();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    // std::cout << "Select: " << elapsed.count() * 1000 << "ms" << std::endl;

    AnchorIndexLog *index_log;
    AnchorScheduleLog *schedule_log;
    if (save_log_)
    {
        for (auto id: target_ids)
        {
            for (auto frame : streams[id]->anchors)
            {
                index_log = new AnchorIndexLog(frame->current_video_frame, frame->current_super_frame, curr_epoch_);
                streams[id]->aaindex_logs.push_back(index_log);
            }
        }
        //    AnchorScheduleLog(double total_budget_, double used_budget_, int count_, int num_anchors_, 
                        // std::chrono::system_clock::time_point start_, std::chrono::system_clock::time_point end_)
 
        schedule_log = new AnchorScheduleLog(budget_in_ms, budget_total - budget_in_ms, curr_epoch_, num_total_anchors, schedule_start, schedule_end);
        schedule_logs_.push_back(schedule_log);
    }

    // free anchors
    for (auto id: target_ids)
    {
        EngorgioStream *stream = stream_context_->streams[id];
        if (neural_enhancer_)
        {
            stream->anchors.clear();
        }
        else
        {
            EngorgioFramePool *framepool = stream->framepool;
            for (auto & anchor : stream->anchors)
                framepool->FreeFrame(anchor);
            stream->anchors.clear();
        }
    }

    // free non-anchors 
    // auto start3 = std::chrono::high_resolution_clock::now();   
    while (engorgio_context_->global_key_frames.size() > 0)
    {
        frame = engorgio_context_->global_key_frames.back();
        engorgio_context_->global_key_frames.pop_back();
        stream = streams[frame->stream_id];
        stream->framepool->FreeFrame(frame);
    }
    while (engorgio_context_->global_altref_frames.size() > 0)
    {
        frame = engorgio_context_->global_altref_frames.top();
        engorgio_context_->global_altref_frames.pop();
        stream = streams[frame->stream_id];
        stream->framepool->FreeFrame(frame);
    }
    while (engorgio_context_->global_normal_frames.size() > 0)
    {
        // std::cout << "normal: " << engorgio_context_->global_normal_frames.size() << std::endl;
        frame = engorgio_context_->global_normal_frames.top();
        engorgio_context_->global_normal_frames.pop();
        stream = streams[frame->stream_id];
        stream->framepool->FreeFrame(frame);
    }
    // auto end3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed3 = end3 - start3;
    // std::cout << "Free: " << elapsed3.count() * 1000 << "ms" << std::endl;

    // deprecated: old log
    // if (save_log_)
    // {
    //     AnchorLog *log = new AnchorLog;
    //     for (auto id: target_ids)
    //     {
    //         log->anchor_indexes.insert(pair<int, std::deque<std::string>>(id, std::deque<std::string>()));
    //         for (auto anchor: streams[id]->anchors)
    //         {
    //             std::string idx = std::to_string(anchor->current_video_frame) + '.' + std::to_string(anchor->current_super_frame);
    //             log->anchor_indexes[id].push_back(idx);
    //         }
    //         log->sort_latency = elapsed1.count() * 1000; 
    //         log->select_latency = elapsed2.count() * 1000;
    //         log->free_latency = elapsed3.count() * 1000;
    //     }
    //     engrogio_logs_.push_back(log);
    // }

    curr_epoch_ += 1;
}

void AnchorEngine::RunPerFrame()
{
    std::vector<EngorgioStream*> streams = stream_context_->streams;
    std::mutex &smutex = stream_context_->mutex;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::vector<int> target_ids;

    smutex.lock();
    for (auto id : active_ids)
    {
        if (streams[id]->frames.size() > 0)
            target_ids.push_back(id);
    }
    smutex.unlock();

    EngorgioStream *stream;
    EngorgioFrame *frame;
    int num_frames;
    for (auto id: target_ids)
    {
        stream = stream_context_->streams[id];
        stream->mutex.lock();
        num_frames = stream->frames.size();
        for (int i = 0; i < num_frames; i++)
        {
            frame = stream->frames.front();
            if (frame->frame_type != 1)
                stream->anchors.push_back(frame);
            else
                stream->framepool->FreeFrame(frame);
            stream->frames.pop_front();
        }
        stream->mutex.unlock();
    }

    if (neural_enhancer_)
    {
        for (auto id: target_ids)
        {
            // std::cout << "Query sents" << std::endl;
            stream = streams[id];
            if (stream->anchors.size() > 0)
            //TODO: interval * 5로...
            neural_enhancer_->Process(id, stream->model, stream->framepool, stream->anchors, false);
            // std::cout << "anchors size: " << stream->anchors.size() << std::endl;
            // stream->anchors.clear(); // TODO: move this after logging
        }
    }

    // free anchors
    for (auto id: target_ids)
    {
        EngorgioStream *stream = stream_context_->streams[id];
        if (neural_enhancer_)
        {
            stream->anchors.clear();
        }
        else
        {
            EngorgioFramePool *framepool = stream->framepool;
            for (auto & anchor : stream->anchors)
                framepool->FreeFrame(anchor);
            stream->anchors.clear();
        }
    }

    // deprecated: old log
    // if (save_log_)
    // {
    //     AnchorLog *log = new AnchorLog;
    //     for (auto id: target_ids)
    //     {
    //         log->anchor_indexes.insert(pair<int, std::deque<std::string>>(id, std::deque<std::string>()));
    //         for (auto anchor: streams[id]->anchors)
    //         {
    //             std::string idx = std::to_string(anchor->current_video_frame) + '.' + std::to_string(anchor->current_super_frame);
    //             log->anchor_indexes[id].push_back(idx);
    //         }
    //     }
    //     engrogio_logs_.push_back(log);
    // }
}

void AnchorEngine::RunSelective()
{
    std::vector<EngorgioStream*> streams = stream_context_->streams;
    std::mutex &smutex = stream_context_->mutex;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::vector<int> target_ids;

    smutex.lock();
    for (auto id : active_ids)
    {
        if (streams[id]->frames.size() > 0)
            target_ids.push_back(id);
    }
    smutex.unlock();

    EngorgioStream *stream;
    EngorgioFrame *frame;
    int num_frames;
    int num_anchors = selective_context_->num_anchors;
    for (auto id: target_ids)
    {
        stream = stream_context_->streams[id];
        stream->mutex.lock();
        num_frames = stream->frames.size();  
        double interval = (double) num_frames / num_anchors;
        for (int i = 0; i < num_frames; i++)
        {
            if ((i % int(ceil(interval))) == 0)
            {
                frame = stream->frames.front();
                stream->anchors.push_back(frame);
                stream->frames.pop_front();
            }
            else
            {
                frame = stream->frames.front();
                stream->frames.pop_front();
                stream->framepool->FreeFrame(frame);
            }
        }
        stream->mutex.unlock();
    }

    // deprecated: old log
    // if (save_log_)
    // {
    //     AnchorLog *log = new AnchorLog;
    //     for (auto id: target_ids)
    //     {
    //         log->anchor_indexes.insert(pair<int, std::deque<std::string>>(id, std::deque<std::string>()));
    //         for (auto anchor: streams[id]->anchors)
    //         {
    //             std::string idx = std::to_string(anchor->current_video_frame) + '.' + std::to_string(anchor->current_super_frame);
    //             log->anchor_indexes[id].push_back(idx);
    //         }
    //     }
    //     engrogio_logs_.push_back(log);
    // }
}

void AnchorEngine::Run()
{
    switch (policy_)
    {
        case SelectPolicy::ENGORGIO_BUDGET:
        case SelectPolicy::ENGORGIO_FRACTION:
            RunEngorgio();
            break;
        case SelectPolicy::PERFRAME:
            RunPerFrame();
            break;
        case SelectPolicy::SELECTIVE:
            RunSelective();
            break;
        default:
            throw std::invalid_argument("Invalid policy");
    } 
}

void AnchorEngine::SaveLog()
{

    switch (policy_)
    {
        case SelectPolicy::ENGORGIO_BUDGET:
        case SelectPolicy::ENGORGIO_FRACTION:
            SaveEngorgioLog();
            break;
        case SelectPolicy::PERFRAME:
            break;
        case SelectPolicy::SELECTIVE:
            break;
        default:
            throw std::invalid_argument("Invalid policy");
    } 
}

void AnchorEngine::SaveLog(int stream_id)
{

    switch (policy_)
    {
        case SelectPolicy::ENGORGIO_BUDGET:
        case SelectPolicy::ENGORGIO_FRACTION:
            SaveEngorgioLog(stream_id);
            break;
        case SelectPolicy::PERFRAME:
            break;
        case SelectPolicy::SELECTIVE:
            break;
        default:
            throw std::invalid_argument("Invalid policy");
    } 
}

void AnchorEngine::RunPeriodic()
{
    periodic_ = true;
    auto start_global = std::chrono::high_resolution_clock::now();
    auto start_epoch = std::chrono::high_resolution_clock::now();
    auto end_epoch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_epoch, elapsed_global;
    int count = 1;
    // int interval_in_ms;
    int interval_in_micros;
    if (policy_ == SelectPolicy::ENGORGIO_BUDGET or policy_ == SelectPolicy::ENGORGIO_FRACTION)
        interval_in_micros = INTERVAL_IN_MICROS; 
        // interval_in_ms = INTERVAL_IN_MS;
    else
        interval_in_micros = INTERVAL_PER_FRAME_IN_MiCROS;
        // interval_in_ms = INTERVAL_PER_FRAME_IN_MS;

    while (periodic_)
    {
        end_epoch = std::chrono::high_resolution_clock::now();
        elapsed_epoch = end_epoch - start_epoch;
        elapsed_global = end_epoch - start_global;
        if (elapsed_epoch.count() * 1000 * 1000 > interval_in_micros)
        {
            std::cout << count << " epoch: " << elapsed_global.count()  << "sec passed" << std::endl;
            count += 1;
            Run();
            // std::cout << elapsed_global.count() << "," << elapsed_epoch.count() << std::endl;
            start_epoch = std::chrono::high_resolution_clock::now();
        }
        
    }
    // std::cout << "RunPeriodic() ends" << std::endl;
}

void AnchorEngine::LaunchPeriodic()
{
    // std::cout << "thread index: " << thread_index_ << std::endl;
    cpu_set_t cpuset;
    periodic_thread_ = new std::thread([this](){this->RunPeriodic();});
    CPU_ZERO(&cpuset);
    CPU_SET(thread_index_, &cpuset);
    int rc = pthread_setaffinity_np(periodic_thread_->native_handle(),
                                sizeof(cpu_set_t), &cpuset);
    assert (rc == 0);
}

void AnchorEngine::StopPeriodic()
{
    // std::cout <<  "StopPeriodic()" << std::endl;
    periodic_ = false;
    if (periodic_thread_)
    {
        periodic_thread_->join();
        delete periodic_thread_;
    }
    // sleep (interval - selection latency) - invoke periodically
    // run_periodic = false
}

// deprecated: oild log
// void AnchorEngine::SaveEngorgioIndexLog()
// {
//     if (!save_log_)
//         return;

//     // set base dir and create it 
//     if (log_dir_.empty())
//     {
//         // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
//         std::string today = date::format("%F", std::chrono::system_clock::now());   
//         // log_dir_ = fs::current_path() / "results" / "anchor_engine" / today;
//         log_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "anchor_engine" / today;
//     }
//     if (!fs::exists(log_dir_))
//         fs::create_directories(log_dir_);
       
//     // save logs
//     fs::path index_path;
//     std::ofstream index_file;
//     std::string index_log;

//     int i = 0; 
//     for (auto log : engrogio_logs_)
//     {
//         index_path = log_dir_ / ("index" + std::to_string(i) + ".txt");
//         index_file.open(index_path);
//         if (index_file.is_open())
//         {
//             index_log = "content\tanchor indexes\n";
//             index_file << index_log;
//             for (auto const & [id, anchor_indexes]: log->anchor_indexes)
//             {
//                 index_file << stream_context_->streams[id]->content << '\t';
//                 for (auto index : anchor_indexes)
//                     index_file << index << " ";
//                 index_file << "\n";
//             }
//         }
//         index_file.close();
    
//         i++;
//     }
// }
// void AnchorEngine::SaveEngorgioLatencyLog()
// {
//     if (!save_log_)
//         return;

//     // set base dir and create it 
//     if (log_dir_.empty())
//     {
//         // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
//         std::string today = date::format("%F", std::chrono::system_clock::now());   
//         // log_dir_ = fs::current_path() / "results" / "anchor_engine" / today;
//         log_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "anchor_engine" / today;
//     }
//     if (!fs::exists(log_dir_))
//         fs::create_directories(log_dir_);
       
//     // save logs
//     fs::path latency_path;
//     std::ofstream latency_file;
//     std::string latency_log;

//     int i;
//     switch (policy_)
//     {
//         case SelectPolicy::ENGORGIO:
//             latency_path = log_dir_ / "latency.txt";
//             latency_file.open(latency_path);
//             if (latency_file.is_open())
//             {
//                 latency_log = "round\tsort latency (ms)\tselect latency (ms)\tfree latency (ms)\n";
//                 latency_file << latency_log;
//             }
                
//             i = 0; 
//             for (auto log : engrogio_logs_)
//             {
//                 if (latency_file.is_open())
//                 {
//                     latency_log = std::to_string(i) + '\t' + std::to_string(log->sort_latency) + 
//                             '\t' + std::to_string(log->select_latency) + '\t' + std::to_string(log->free_latency) + '\n'; 
//                     latency_file << latency_log;
//                 }
                
//                 i++;
//             }

//             latency_file.close();        
//             break;
//         default:
//             break;
//     }

// }

void AnchorEngine::SaveEngorgioLog()
{
    if (!save_log_)
        return;

    // set base dir and create it
    std::filesystem::path log_dir; 
    std::string today = date::format("%F", std::chrono::system_clock::now());   
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
        // log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / today;
        log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results";
    }
    else
    {
        // log_dir = log_dir_ / today;
        log_dir = log_dir_;
    }
    if (!fs::exists(log_dir))
        fs::create_directories(log_dir);

    // std::cout << log_dir << std::endl;
       
    // anchor index log
    fs::path log_path;
    std::ofstream log_file;
    std::chrono::duration<double> start_elapsed, end_elapsed;
    log_path = log_dir / "anchor_latency.txt";
    log_file.open(log_path);
    if (log_file.is_open())
    {
        log_file << "Total budget (ms)\tUsed budget (ms)\t#Anchors\tEpoch\tStart (s)\tEnd (s)" << '\n';
        for (auto log : schedule_logs_)
        {
            start_elapsed = log->start - start_;
            end_elapsed = log->end - start_;
            log_file << std::to_string(log->total_budget) << "\t"
                         << std::to_string(log->used_budget) << "\t"
                         << std::to_string(log->num_anchors) << "\t"
                         << std::to_string(log->count) << "\t"
                         << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\n";
        }
    }
    log_file.flush();
    log_file.close();
}

void AnchorEngine::SaveEngorgioLog(int stream_id)
{
    if (!save_log_)
        return;

    // set base dir and create it 
    std::filesystem::path log_dir;
    std::string today = date::format("%F", std::chrono::system_clock::now());   
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
        log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / today / std::to_string(stream_id);
    }
    else
    {
        log_dir = log_dir_ / today / std::to_string(stream_id);
    }
    if (!fs::exists(log_dir))
        fs::create_directories(log_dir);
       
    // anchor index log
    fs::path log_path;
    std::ofstream log_file;
    EngorgioStream *stream = stream_context_->streams[stream_id];
    log_path = log_dir / "anchor_index.txt";
    log_file.open(log_path);
    if (log_file.is_open())
    {
        log_file << "Video index\tSuper index" << '\n';
        for (auto log : stream->aaindex_logs)
        {
            log_file << std::to_string(log->video_index) << "\t"
                         << std::to_string(log->super_index) << "\t"
                         << std::to_string(log->curr_epoch) << "\n";
        }
    }
    log_file.close();

    // frame index log
    log_path = log_dir / "frame_index.txt";
    log_file.open(log_path);
    if (log_file.is_open())
    {
        log_file << "Video index\tSuper index" << '\n';
        for (auto log : stream->afindex_logs)
        {
            log_file << std::to_string(log->video_index) << "\t"
                        << std::to_string(log->super_index) << "\t"
                         << std::to_string(log->count) << "\n";
        }
    }
    log_file.flush();
    log_file.close();
}

void AnchorEngine::LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer)
{
    neural_enhancer_ = neural_enhancer;
}

void AnchorEngine::LoadDNNLatency(int resolution, std::string &name, double latency)
{
    std::pair<int, std::string> key = std::make_pair(resolution, name);
    if (dnn_latencies_.find(key) == dnn_latencies_.end())
    {
        dnn_latencies_[key] = latency;
    }
}

int AnchorEngine::GetTotalAnchors()
{
    return num_total_anchors_;
}
