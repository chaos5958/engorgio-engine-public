#include "decode_engine.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include "vpx_scale/yv12config.h"
#include "date/date.h"
#include "ipp.h"
#include "ippcc.h"
#include "ippi.h"
#include "enhancer_common.h"

using namespace std;

DecodeEngine::DecodeEngine(DecodeEngineProfile &profile, EngorgioStreamContext *stream_context)
{
    num_workers_ = profile.num_workers;
    save_log_ = profile.save_log;
    start_ = profile.start;
    stream_context_ = stream_context;
    log_dir_ = profile.log_dir;
    
    std::vector<int> thread_indexes;
    thread_indexes = profile.thread_indexes;

    // init, launch workers
    cpu_set_t cpuset;
    int rc;
    for (int i = 0; i < num_workers_; i++)
    {
        struct DecodeWorker* worker = new DecodeWorker;
        worker->thread = new std::thread([this, worker, i](){this->DecodeEventHandler(*worker, i);});
        workers_.push_back(worker);
        CPU_ZERO(&cpuset);
        CPU_SET(thread_indexes[i], &cpuset);
        rc = pthread_setaffinity_np(worker->thread->native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
        assert (rc == 0);
    }
}

// DecodeEngine::DecodeEngine(int num_workers, std::vector<Stream*> &streams): stream_context_->streams(streams)
// {
//     num_workers_ = num_workers;
//     Build();
// }

DecodeEngine::~DecodeEngine()
{
    // stop, destroy workers
    DecodeEvent event;
    event.type = DecodeType::kJoin;
    for (auto worker : workers_)
    {
        // cout << worker->num_streams << endl;
        worker->mutex.lock();
        worker->events.push_back(event);
        worker->mutex.unlock();
    }

    for (auto worker : workers_)
    {
        worker->thread->join();
        delete worker->thread;
        delete worker;
    }
}

void DecodeEngine::DecodeEventHandler(DecodeWorker &worker, int index)
{
    DecodeEvent event;
    bool has_event = false;

    while (1)
    {
        worker.mutex.lock();
        has_event = false;
        if (!worker.events.empty())
        {
            event = worker.events.front();
            worker.events.erase(worker.events.begin());
            has_event = true;
        }
        worker.mutex.unlock();

        if (has_event)
        {
            switch (event.type)
            {
            case DecodeType::kInit:
                DecoderInit(event);
                break;
            case DecodeType::kDecode:
                DecoderDecode(event, index);
                break;
            case DecodeType::kDestroy:
                DecoderDestroy(event);
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

void DecodeEngine::DecoderInit(DecodeEvent &event)
{
    auto start = std::chrono::high_resolution_clock::now();
    int stream_id = event.stream_id;

    vpx_codec_ctx_t* decoder  = new vpx_codec_ctx_t();
    vpx_codec_dec_cfg cfg = {0, 0, 0};

    vpx_codec_err_t err = vpx_codec_dec_init(decoder, &vpx_codec_vp9_dx_algo, &cfg, 0);
    if (err)
    {
        cerr << "Failed to initialize libvpx decoder, error =" << err << endl;
        return;
    }

#ifdef VPX_CTRL_VP9_DECODE_SET_ROW_MT
    err = vpx_codec_control(decoder, VP9D_SET_ROW_MT, enable_multi_thread_);
    if (err)
    {
        cerr << "Failed to enable row multi thread mode, error = " << err << endl;
    }
#endif

    if (disable_loop_filter_)
    {
        err = vpx_codec_control(decoder, VP9_SET_SKIP_LOOP_FILTER, true);
        if (err)
        {
            cerr << "Failed to shut off libvpx loop filter, error = " << err << endl;
        }
    }
#ifdef VPX_CTRL_VP9_SET_LOOP_FILTER_OPT
    else
    {
        err = vpx_codec_control(decoder, VP9D_SET_LOOP_FILTER_OPT, true);
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
    stream_context_->streams[stream_id]->decoder = decoder;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Init): " << elapsed.count() * 1000 << "ms" << std::endl;
    // cout << "stream" << stream_id << "is initialized" << endl;
}

// static void print_frame(EngorgioFrame *frame)
// {
//     if (frame == nullptr)
//         return;

//     std::cout << "index: " << frame->current_video_frame << ',' << frame->current_super_frame << '\t';
//     std::cout << "size : " << frame->residual_size << std::endl;
// }

// static void write_file(std::string path, Ipp8u *data, int len)
// {
//       std::ofstream fout;
// 	fout.open(path, std::ios::out | std::ios::binary);
    
// 	if (fout.is_open()){
// 		fout.write((const char*)data, len);
// 		fout.close();
// 	}
// }

static EngorgioFrame* copy_frame(int stream_id, EngorgioStream *stream, YV12_BUFFER_CONFIG *src_frame, bool is_visible)
{
    if (src_frame == nullptr)
        return nullptr;

    // allocate a frame 
    EngorgioFrame *frame = stream->framepool->GetFrame(src_frame->y_crop_width, src_frame->y_crop_height);
    frame->stream_id = stream_id;

    // run YUV2RGB conversion
    const Ipp8u* yuv_buffer[3];
    yuv_buffer[0] = src_frame->y_buffer;
    yuv_buffer[1] = src_frame->u_buffer;
    yuv_buffer[2] = src_frame->v_buffer;
    int yuv_stride[3] = {src_frame->y_stride, src_frame->uv_stride, src_frame->uv_stride};
    int rgb_stride = frame->width * 3;
    IppiSize roi_size = {frame->width, frame->height};
    const int rgb_order[3] = {2, 1, 0};

    // auto start = std::chrono::high_resolution_clock::now();
    IppStatus st = ippStsNoErr;      
    st = ippiYCbCr420ToBGR_709CSC_8u_P3C3R(yuv_buffer, yuv_stride,frame->rgb_buffer, rgb_stride, roi_size);
    if ( st != ippStsNoErr)
    {
        std::cout << "failed: " << st << std::endl;
        return nullptr;
    }
    st = ippiSwapChannels_8u_C3R(frame->rgb_buffer, rgb_stride, frame->rgb_buffer, rgb_stride, roi_size, rgb_order);
    if ( st != ippStsNoErr)
    {
        std::cout << "failed: " << st << std::endl;
        return nullptr;
    }

    // For, perforamnce measurement (TODO: offload rgb-yuv conversion to GPUs)
    // int num_pixels = frame->width * frame->height;
    // memcpy(frame->rgb_buffer, yuv_buffer[0], num_pixels);
    // memcpy(frame->rgb_buffer + num_pixels, yuv_buffer[1], num_pixels / 4);
    // memcpy(frame->rgb_buffer + num_pixels * 2, yuv_buffer[2], num_pixels / 4);
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (YUV2RGB): " << elapsed.count() * 1000 << "ms" << std::endl;

    // save rgb frames (validation)          
    // std::string img_path = std::to_string(src_frame->current_video_frame) + "_" + std::to_string(src_frame->current_super_frame) + ".rgb";  
    // write_file(img_path, frame->rgb_buffer, frame->width * frame->height * 3);

    // set frame metadata
    frame->current_video_frame = src_frame->current_video_frame;
    frame->current_super_frame = src_frame->current_super_frame;
    frame->residual_size = src_frame->residual_size;
    if (!is_visible && src_frame->frame_type == 2)
        frame->frame_type = 1;
    else
        frame->frame_type = src_frame->frame_type;

    return frame;
}

void DecodeEngine::DecoderDecode(DecodeEvent &event, int index)
{
    // std::cout << "Thread #" << index << ": on CPU " << sched_getcpu() << "\n";
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    int stream_id = event.stream_id;
    uint8_t *buf = event.buf;
    int len = event.len;
    EngorgioFrame *visible_frame = nullptr;
    EngorgioFrame *non_visible_frame = nullptr;
    vpx_image_pair_t *img_pair;
    vpx_codec_iter_t iter = NULL;
    EngorgioStream *stream = stream_context_->streams[stream_id];

    // const vpx_codec_err_t status = vpx_codec_decode(stream->decoder, NULL, 0, NULL, 0);
    const vpx_codec_err_t status = vpx_codec_decode(stream->decoder, buf, len, NULL, 0);
    if (status != VPX_CODEC_OK)
    {
        cerr << "vpx_codec_decode() failed, status = " << status << endl;
    }

    img_pair = vpx_codec_get_frames(stream->decoder, &iter);
    if (img_pair == nullptr)
    {
        cerr << "vpx_codec_get_frames() failed" << endl;
    }

    int curr_size = 0, last_size;
    if (img_pair->non_visible != nullptr)
    {
        // std::cout << "invisible frame" << std::endl;
        non_visible_frame = copy_frame(stream_id, stream, img_pair->non_visible, false);
        curr_size += non_visible_frame->residual_size;
    }        
    visible_frame = copy_frame(stream_id, stream, img_pair->visible, true);
    curr_size += visible_frame->residual_size;
 
    if (stream->sizes.size() == (std::size_t) stream->gop * 2)
    {
        last_size = stream->sizes.front();
        stream->avg_size = (stream->sizes.size() * stream->avg_size - last_size + curr_size) / stream->sizes.size();
        stream->sizes.pop_front();
        stream->sizes.push_back(curr_size);
    }   
    else
    {
        stream->avg_size = (stream->sizes.size() * stream->avg_size + curr_size) / (stream->sizes.size()+ 1);
        stream->sizes.push_back(curr_size);
    }
    // std::cout << "avg_size: " << stream->avg_size << std::endl;
    // if (visible_frame->current_video_frame < 10)
    // {
    //     std::cout << "bitrate: " << stream->avg_size * 60 * 8 / 1024 << "," << curr_size <<  std::endl;
    // }
    //      avg = (N*avg-queue.dequeue()+num)/N
    // else
    //      avg = (queue.size()*avg+num)/(queue.size()+1)
    //      queue.add(num);

    // auto decode_end = std::chrono::high_resolution_clock::now();
    // visible_frame->decode_start =  decode_start;
    // visible_frame->decode_end =  decode_end;
    // if (img_pair->non_visible != nullptr)
    // {
    //     non_visible_frame->decode_start = decode_start;
    //     non_visible_frame->decode_end = decode_end;
    // }

    stream->mutex.lock();
    if (img_pair->non_visible != nullptr)
        stream->frames.push_back(non_visible_frame);
    stream->frames.push_back(visible_frame);
    stream->mutex.unlock();

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

    DecodeLatencyLog *log;
    if (save_log_)
    {
        log = new DecodeLatencyLog(visible_frame->current_video_frame, visible_frame->current_super_frame,
                                    start, end);
        stream->dlatency_logs.push_back(log);
        if (non_visible_frame)
        {
            log = new DecodeLatencyLog(non_visible_frame->current_video_frame, non_visible_frame->current_super_frame,
                                    start, end);
            stream->dlatency_logs.push_back(log);
        }               
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Decode " << stream_context_->streams[stream_id]->num_frames << "): " << elapsed.count() * 1000 << "ms" << std::endl;    
}

void DecodeEngine::DecoderDestroy(DecodeEvent &event)
{
    // auto start = std::chrono::high_resolution_clock::now();

    int stream_id = event.stream_id;
    int idx = stream_context_->streams[stream_id]->worker_index;
    vpx_codec_destroy(stream_context_->streams[stream_id]->decoder);
    stream_context_->streams[stream_id]->decoder = nullptr;

    mutex_.lock();
    workers_[idx]->num_streams-= 1; // TODO: here
    mutex_.unlock();

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Latency (Destroy): " << elapsed.count() * 1000 << "ms" << std::endl;
}


// TODO: increase, decrease here
int DecodeEngine::DecoderInit(int stream_id)
{
    // select a thread which num_streams is minimal 
    // TODO: validate this 
    int idx, selected_idx, num_streams = -1;
    mutex_.lock();
    idx = 0;
    for (auto &worker: workers_)
    {
        // std::cout << worker->num_streams << "," << num_streams << std::endl;
        if (num_streams == -1 or worker->num_streams < num_streams)
        {
            selected_idx = idx;
            num_streams = worker->num_streams;
        }
        idx += 1;
    }
    stream_context_->streams[stream_id]->worker_index = selected_idx;
    workers_[selected_idx]->num_streams += 1;
    mutex_.unlock();

    // // cout << workers_.size() << '\t' << num_workers_ << endl;
    // cout << "stream id: " << stream_id << ", selected_idx is " << selected_idx << endl;
    // cout << "workers: " << workers_.size() << endl;
    // cout << "workers: " << workers_[selected_idx]->num_streams<< endl;
        
    // 2. push an event to a thread
    DecodeEvent event;
    event.type = DecodeType::kInit;
    event.stream_id = stream_id;
    DecoderInit(event);
    // workers_[selected_idx]->mutex.lock();
    // workers_[selected_idx]->events.push_back(event);
    // workers_[selected_idx]->mutex.unlock();
    // cout << "events length: " << workers_[selected_idx]->events.size() << endl;
    return 0;
}

int DecodeEngine::DecoderDestroy(int stream_id)
{
    // cout << stream_id << "is destroyed, " << workers_[stream_context_->streams[stream_id]->worker_index]->events.size() << endl;
    // find a thread
    int idx = stream_context_->streams[stream_id]->worker_index;
        
    // push an event to a thread
    DecodeEvent event;
    event.type = DecodeType::kDestroy;
    event.stream_id = stream_id;
    workers_[idx]->mutex.lock();
    workers_[idx]->events.push_back(event);
    workers_[idx]->mutex.unlock();

    return 0;
}

int DecodeEngine::DecoderDecode(int stream_id, uint8_t *buf, int len)
{
    // find a thread
    int idx = stream_context_->streams[stream_id]->worker_index;
    
    // push an event to a thread
    // TODO: do we need a lockless queue here? 
    DecodeEvent event;
    event.type = DecodeType::kDecode;
    event.stream_id = stream_id;
    event.buf = buf;
    event.len = len;
    workers_[idx]->mutex.lock();
    workers_[idx]->events.push_back(event);
    workers_[idx]->mutex.unlock();

    return 0;
}

void DecodeEngine::SaveLog(int stream_id)
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
        // log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / today / std::to_string(stream_id);
        log_dir = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / std::to_string(stream_id);
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

    EngorgioStream *stream = stream_context_->streams[stream_id];
    latency_path = log_dir / "decode_latency.txt";
    latency_file.open(latency_path);
    if (latency_file.is_open())
    {
        latency_file << "Video index\tSuper index\tStart(s)\tEnd(s)" << '\n';
        std::chrono::duration<double> start_elapsed, end_elapsed, latency;
        for (auto log : stream->dlatency_logs)
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

    latency_path = log_dir / "stream.txt";
    latency_file.open(latency_path);
    if (latency_file.is_open())
    {
        latency_file << "Content\t" << stream->content << "\n";
        if (stream->model)
            latency_file << "Model\t" << stream->model->name_ << "\n";
    }
    latency_file.flush();
    latency_file.close();
}

// deprecated: old log
// void DecodeEngine::SaveLog()
// {
//     if (!save_log_)
//         return;

//     // set base dir and create it 
//     if (log_dir_.empty())
//     {
//         // default dir:  ([binary dir]/results/anchor_engine/[year-month-date]/...)
//         std::string today = date::format("%F", std::chrono::system_clock::now());   
//         // log_dir_ = fs::current_path() / "results" / "decode_engine" / today;
//         log_dir_ = std::filesystem::path("/workspace") / "research" / "engorgio-engine" / "results" / "decode_engine" / today;
//     }
//     if (!fs::exists(log_dir_))
//         fs::create_directories(log_dir_);
       
//     // save logs
//     fs::path latency_path;
//     std::ofstream latency_file;
//     std::string latency_log;

//     std::chrono::duration<double> elapsed;
//     double throughput;
//     DecodeWorker *worker;
//     latency_path = log_dir_ / "latency.txt";
//     latency_file.open(latency_path);
//     if (latency_file.is_open())
//     {
//         for (int i = 0; i < workers_.size(); i++)
//         {
//             worker = workers_[i];
//             elapsed = worker->end - worker->start;
//             throughput = worker->num_frames / elapsed.count(); // require limit
//             latency_file << "Worker" << to_string(i) << '\t';
//             latency_file << to_string(throughput) << "fps" << '\n';
//         }
//     }
//     latency_file.close();
// }