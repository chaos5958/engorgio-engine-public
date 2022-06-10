#include "date/date.h"
#include "enhancer_common.h"
#include "encode_engine.h"
#include "control_common.h"
#include "cuHostMemoryV2.h"
#include <fstream>
#include <filesystem>
#include <turbojpeg.h>

// #include "kdu_functions.h"


namespace fs = std::filesystem;

const int INIT_NUM_JPEG_FRAMES = 100;
const int INIT_WIDTH = 3840;
const int INIT_HEIGHT = 2160;



JPEGEncodeEngine::JPEGEncodeEngine(JPEGProfile &profile)
{
    encode_count_ = 0;
    save_count_ = 0;
    num_encode_workers_ = profile.num_encode_workers_;
    num_save_workers_ = 1;
    save_image_ = profile.save_image_;
    save_log_ = profile.save_log_;
    log_dir_ = profile.log_dir_;
    start_ = profile.start_;
    thread_indexes_ = profile.thread_indexes_;

    // codec parameters
    subsampling_ = profile.subsampling_;
    qp_ = profile.qp_;
    flag_ = profile.flag_;

    // frame pool
    framepool_ = new JPEGFramePool(subsampling_, INIT_NUM_JPEG_FRAMES, INIT_WIDTH, INIT_HEIGHT);

    // launch threads
    // TODO: CPU pinning
    cpu_set_t cpuset;
    for (int i = 0; i < num_encode_workers_; i++)
    {
        encode_workers_.push_back(new std::thread([this, i](){this->EncodeHandler(i);}));
        CPU_ZERO(&cpuset);
        // CPU_SET(30, &cpuset);
        // CPU_SET(31, &cpuset);
        // CPU_SET(32, &cpuset);
        // CPU_SET(33, &cpuset);
        CPU_SET(thread_indexes_[i], &cpuset);
        int rc = pthread_setaffinity_np(encode_workers_[i]->native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
        assert(rc == 0);
    }

    assert(num_save_workers_ == 1);
    for (int i = 0; i < num_save_workers_; i++)
    {
        save_workers_.push_back(new std::thread([this](){this->SaveImageHandler(0);}));
        CPU_ZERO(&cpuset);
        CPU_SET(thread_indexes_[num_encode_workers_], &cpuset);
        int rc = pthread_setaffinity_np(save_workers_[i]->native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
        assert(rc == 0);
    }

    // log
    for (int i = 0; i < num_encode_workers_; i++)
    {
        logs_.emplace_back(std::deque<JPEGQueryLog*>());
    }
}

JPEGEncodeEngine::~JPEGEncodeEngine()
{
    // logging
    SaveLog();

    // destroy threads
    JPEGEncodeEvent event;
    event.type_ = JPEGEventType::kJoin;
    for (int i = 0; i < num_encode_workers_; i++)
    {
        encode_mutex_.lock();
        encode_events_.push_back(event);
        encode_mutex_.unlock();
        
    }
    for (int i = 0; i < num_encode_workers_; i++)
    {
        encode_workers_[i]->join();
        delete encode_workers_[i];
    }

    JPEGSaveEvent sevent;
    sevent.type_ = JPEGEventType::kJoin;
    for (int i = 0; i < num_save_workers_; i++)
    {
        save_mutex_.lock();
        save_events_.push_back(sevent);
        save_mutex_.unlock();
        
    }
    for (int i = 0; i < num_save_workers_; i++)
    {
        save_workers_[i]->join();
        delete save_workers_[i];
    }

    // destroy framepool
    delete framepool_;
}

void JPEGEncodeEngine::EncodeHandler(int index)
{
    JPEGEncodeEvent event;
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
            case JPEGEventType::kEncode:
                Encode(index, event);

                break;
            case JPEGEventType::kEncodeKDU:
                Encode_kakadu(index, event);

                break;
            case JPEGEventType::kJoin:
                return;
            default:
                std::cerr << "Unsupported event type" << std::endl;
                break;
            }
        }
    }
}

void JPEGEncodeEngine::SaveImageHandler(int index)
{
    JPEGSaveEvent event;
    bool has_event = false;

    while (1)
    {
        save_mutex_.lock();
        has_event = false;
        if (!save_events_.empty())
        {
            event = save_events_.front();
            save_events_.pop_front();
            has_event = true;
        }
       save_mutex_.unlock();

        if (has_event)
        {
            // std::cout << "unload event" << std::endl;
            switch (event.type_)
            {
            case JPEGEventType::kSave:
                SaveImage(event);
                break;
            case JPEGEventType::kJoin:
                return;
            default:
                std::cerr << "Unsupported event type" << std::endl;
                break;
            }
        }
    }
}

void JPEGEncodeEngine::Encode(int stream_id, std::vector<EngorgioFrame*> &frames)
{
    // std::cout << "Encode: " << frames.size() << "," << frames[0]->width << "," << frames[1]->height << std::endl;
    JPEGEncodeEvent event = {JPEGEventType::kEncode, stream_id, frames};
    encode_mutex_.lock();
    encode_events_.push_back(event);
    encode_mutex_.unlock();
}


void JPEGEncodeEngine::Encode_kakadu(int stream_id, std::vector<EngorgioFrame*> &frames)
{
    // std::cout << "Encode: " << frames.size() << "," << frames[0]->width << "," << frames[1]->height << std::endl;
    JPEGEncodeEvent event = {JPEGEventType::kEncodeKDU, stream_id, frames};
    encode_mutex_.lock();
    encode_events_.push_back(event);
    encode_mutex_.unlock();
}


void JPEGEncodeEngine::Encode_kakadu(int index, JPEGEncodeEvent &event)
{
    // auto start = std::chrono::high_resolution_clock::now();

    // tjhandle encodeInstance;
    // std::vector<EngorgioFrame*> &frames = event.frames_;
    // std::deque<JPEGFrame*> jpeg_frames;
    // JPEGFrame* jpeg_frame;
    // JPEGQueryLog *log;
    // cuHostMemoryV2* host_memory = cuHostMemoryV2::GetInstance();

    // int lossless = 0;
    // std::string qp_str_ = std::string("Qfactor=") + std::to_string(qp_);
    // char qp_str[100];
    // memset(qp_str, 0, sizeof(qp_str));
    // strcpy(qp_str, qp_str_.c_str());
    // kdu_byte *j2k_compressed;

    // // std::cout << "[KAKADU]Encode: here1" << std::endl;
    // // std::cout << frames[0]->width << std::endl;

    // if (save_log_)
    //     log = new JPEGQueryLog();
    
    // for (std::size_t i = 0; i < frames.size(); i++)
    // {
    //     // std::cout << frames[i]->width << "," << frames[i]->height << std::endl;

    //     auto start = std::chrono::system_clock::now();
    //     jpeg_frame = framepool_->GetFrame(frames[i]->width, frames[i]->height);
    //     jpeg_frame->curr_video_frame_ = frames[i]->current_video_frame;
    //     jpeg_frame->curr_super_frame_ = frames[i]->current_super_frame;
    //     ht_jp2_encoder encoder;
    //     encoder.init( 3, //num_components
    //                   frames[i]->height,
    //                   frames[i]->width,
    //                   1 // num_threads
    //                   );
    //     j2k_compressed = encoder.encode_memory( frames[i]->rgb_buffer, 
    //                             "", // output file name incase saving is enabled
    //                             lossless, // lossless
    //                             &jpeg_frame->jpg_file_len_,
    //                             qp_str
    //                             // "Qfactor=70" // ex. should put like "Qfactor=85"
    //                             );
    //     free(j2k_compressed);
    //     auto end = std::chrono::system_clock::now();

    //     framepool_->FreeFrame(jpeg_frame);

    //     if (save_log_)
    //     {
    //         log->stream_id = event.stream_id_;
    //         log->video_indexes.push_back(frames[i]->current_video_frame);
    //         log->super_indexes.push_back(frames[i]->current_super_frame);
    //         log->starts.push_back(start);
    //         log->ends.push_back(end);
    //     }
    //     // std::chrono::duration<double> elapsed = end - start;
    //     // std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
    // }

    // if (save_log_)
    //     logs_[index].push_back(log);

    // // free sr frames
    // for (std::size_t i = 0; i < frames.size(); i++)
    // {
    //     if (host_memory) // case 1: given by the infer engine
    //     {
    //         host_memory->Free(frames[i]->height, (void*)frames[i]->rgb_buffer);
    //         frames[i]->rgb_buffer = nullptr;
    //         delete frames[i];
    //     }
    //     else // case 2: given by the tester 
    //     {
    //         delete frames[i];
    //     }
    // }

    // encode_count_mutex_.lock();
    // encode_count_ += frames.size();
    // encode_count_mutex_.unlock();

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
}

void JPEGEncodeEngine::Encode(int index, JPEGEncodeEvent &event)

{
    auto start = std::chrono::high_resolution_clock::now();

    tjhandle encodeInstance;
    std::vector<EngorgioFrame*> &frames = event.frames_;
    std::deque<JPEGFrame*> jpeg_frames;
    JPEGFrame* jpeg_frame;
    JPEGQueryLog *log;
    cuHostMemoryV2* host_memory = cuHostMemoryV2::GetInstance();

    // std::cout << "Encode: here1" << std::endl;
    // std::cout << frames[0]->width << std::endl;

    if (save_log_)
        log = new JPEGQueryLog();
    
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        // std::cout << frames[i]->width << "," << frames[i]->height << std::endl;
        auto start = std::chrono::system_clock::now();
        jpeg_frame = framepool_->GetFrame(frames[i]->width, frames[i]->height);
        jpeg_frame->curr_video_frame_ = frames[i]->current_video_frame;
        jpeg_frame->curr_super_frame_ = frames[i]->current_super_frame;
        encodeInstance = tjInitCompress();
        tjCompress2(encodeInstance, frames[i]->rgb_buffer, frames[i]->width, 0, frames[i]->height, TJPF_RGB, \
        &jpeg_frame->encoded_buffer_, &jpeg_frame->jpg_file_len_, subsampling_, qp_, flag_);
        tjDestroy(encodeInstance);
        auto end = std::chrono::system_clock::now();
    
        if (save_image_)
            jpeg_frames.push_back(jpeg_frame);
        else
            framepool_->FreeFrame(jpeg_frame);

        if (save_log_)
        {
            log->stream_id = event.stream_id_;
            log->video_indexes.push_back(frames[i]->current_video_frame);
            log->super_indexes.push_back(frames[i]->current_super_frame);
            log->starts.push_back(start);
            log->ends.push_back(end);
        }
    }

    if (save_log_)
        logs_[index].push_back(log);

    // log: images
    if (save_image_)
    {
        JPEGSaveEvent sevent = {JPEGEventType::kSave, event.stream_id_, jpeg_frames};
        save_mutex_.lock();
        save_events_.push_back(sevent);
        save_mutex_.unlock();
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

    encode_count_mutex_.lock();
    encode_count_ += frames.size();
    encode_count_mutex_.unlock();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
}

void JPEGEncodeEngine::SaveImage(JPEGSaveEvent &event)
{
    std::deque<JPEGFrame*> &frames = event.frames_;
    JPEGFrame *frame;
    std::string file_name;
    std::filesystem::path file_path;
    std::string today = date::format("%F", std::chrono::system_clock::now());  
    
    std::filesystem::path log_dir; 
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
    log_dir = log_dir / std::to_string(event.stream_id_);
    if (!fs::exists(log_dir))
        fs::create_directories(log_dir);

    std::ofstream fout;
    // std::cout << save_count_ << std::endl;
    // std::cout << frames.size() << std::endl;
    while(frames.size() > 0)
    {
        frame = frames.front();
        frames.pop_front();
        
        file_name = std::to_string(frame->curr_video_frame_) + "_"  + std::to_string(frame->curr_super_frame_) + ".jpg";
        file_path = log_dir / file_name;
        // std::cout << file_path << std::endl;
        // std::cout << file_path << std::endl;
        fout.open(file_path, std::ios::out | std::ios::binary);
        
        if (fout.is_open()){
            fout.write((const char*)frame->encoded_buffer_, frame->jpg_file_len_);
            fout.close();
        }

        framepool_->FreeFrame(frame);

        save_count_mutex_.lock();
        save_count_ += 1;
        save_count_mutex_.unlock();
    }
    // std::cout << save_count_ << std::endl;
}

bool JPEGEncodeEngine::EncodeFinished(int num_frames)
{
    // std::cout << num_frames << "," << encode_count_ << std::endl;
    if (num_frames == encode_count_)
        return true;
    else
        return false;
}

bool JPEGEncodeEngine::SaveFinished(int num_frames)
{
    if (!save_image_)
        return true;

    // std::cout << num_frames << "," << save_count_ << std::endl;
    if (num_frames == save_count_)
        return true;
    else
        return false;
}
struct less_than_key
{
    inline bool operator() (const JPEGFrameLog *log1, const JPEGFrameLog *log2)
    {
        if (log1->video_index != log2->video_index)
            return (log1->video_index < log2->video_index);
        else
            return (log1->super_index < log2->super_index);
    }
};

void JPEGEncodeEngine::SaveLog()
{
    // std::cout << "here1" << std::endl;
    if (!save_log_)
        return;
    // std::cout << "here2" << std::endl;

    std::map<int, std::vector<JPEGFrameLog*>> frame_logs;

    // Build frame-level logs 
    JPEGFrameLog *frame_log;
    for (int i = 0; i < num_encode_workers_; i++)
    {
        for (auto query_log : logs_[i])
        {
            int stream_id = query_log->stream_id;
            if (frame_logs.find(stream_id) == frame_logs.end())
            {
                frame_logs[stream_id] = std::vector<JPEGFrameLog*>();
            }
            for (std::size_t j = 0; j < query_log->video_indexes.size(); j++)
            {
                // query_log->video_indexes[j] = j;
                frame_log = new JPEGFrameLog(query_log, j);
                frame_logs[stream_id].push_back(frame_log);
            }
            delete query_log;
        }
    }

    // Sort frame-level logs
    for (auto & [id, logs] : frame_logs)
    {
        std::sort(logs.begin(), logs.end(), less_than_key());
    }

    // set base dir and create it
    std::filesystem::path base_log_dir, log_dir; 
    std::string today = date::format("%F", std::chrono::system_clock::now());   
    if (log_dir_.empty())
    {
        // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
        // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
        // base_log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / today;
        base_log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results";
    }
    else
    {
        // base_log_dir = log_dir_ / today;
        base_log_dir = log_dir_;
    }
      
    // anchor index log
    fs::path log_path;
    std::ofstream log_file;
    std::chrono::duration<double> start_elapsed, end_elapsed;

    for (auto & [id, logs] : frame_logs)
    {
        log_dir = base_log_dir / std::to_string(id);
        if (!fs::exists(log_dir))
            fs::create_directories(log_dir);

        log_path = log_dir / "encode_latency.txt";
        log_file.open(log_path);
        if (log_file.is_open())
        {
            log_file << "Video index\tSuper index\t" 
                    <<  "Encode (start)\tEncode (end)\t"
                    <<  "\n";
            for (auto log : logs)
            {
                log_file << std::to_string(log->video_index) << "\t"
                         << std::to_string(log->super_index) << "\t";

                start_elapsed = log->start - start_;
                end_elapsed = log->end - start_;
                log_file << std::to_string(start_elapsed.count()) << "\t"
                         << std::to_string(end_elapsed.count()) << "\t";
                log_file << "\n";
            }
        }
        log_file.flush();
        log_file.close();
    }

    // Free frame-level logs
    for (auto & [id, logs] : frame_logs)
    {
        for (auto log: logs)
            delete log;
    }
}
