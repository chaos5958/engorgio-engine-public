#include <iostream>
#include <chrono>
#include <thread>
#include "controller.h"
#include "enhancer_common.h"
using namespace std;

// Controller::Controller(int num_workers): decode_engine_(num_workers, stream_context_->streams)
// {
//     Build();
// }

Controller::Controller(DecodeEngineProfile &dprofile, AnchorEngineProfile &aprofile)
{
    stream_context_ = new EngorgioStreamContext(MAX_NUM_STREAMS);
    decode_engine_ = new DecodeEngine(dprofile, stream_context_);
    anchor_engine_ = new AnchorEngine(aprofile, stream_context_);
}

Controller::~Controller()
{
    delete decode_engine_;
    delete anchor_engine_;
    delete stream_context_;
}

// TODO: handle gop
// TODO: pass per-stream information
int Controller::Init(int gop, std::string &content, EngorgioModel *model)
{
    std::vector<EngorgioStream*> &streams = stream_context_->streams;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::deque<int> &free_ids = stream_context_->free_ids;
    std::mutex &mutex = stream_context_->mutex;
    int stream_id; 

    mutex.lock();
    stream_id = free_ids.front();
    free_ids.pop_front();
    active_ids.push_back(stream_id);
    mutex.unlock();

    streams[stream_id] = new EngorgioStream(gop, content, model);
    decode_engine_->DecoderInit(stream_id);

    return stream_id;
}

void Controller::Process(int stream_id, uint8_t *buf, int len)
{
    decode_engine_->DecoderDecode(stream_id, buf, len);
}


void Controller::Free(int stream_id)
{
    decode_engine_->DecoderDestroy(stream_id);
}

void Controller::Free()
{
    std::vector<EngorgioStream*> &streams = stream_context_->streams;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::deque<int> &free_ids = stream_context_->free_ids;
    std::mutex &mutex = stream_context_->mutex;
    EngorgioStream *stream;


    auto it = active_ids.begin();
    // std::cout << "Controller: Number of active ids: " << active_ids.size() << std::endl;
    
    // std::cout << (it != active_ids.end()) << std::endl;
    while (it != active_ids.end())
    {
        stream = streams[*it];
        while(!(stream->decoder == nullptr && stream->frames.size() == 0 && stream->anchors.size() == 0))
        {
        }
        
        // logging
        decode_engine_->SaveLog(*it);
        anchor_engine_->SaveLog(*it);
        delete stream;
        mutex.lock();
        // std::cout << "Controller: stream" << *it << " is removed"<< std::endl;
        free_ids.push_back(*it);
        it = active_ids.erase(it);
        mutex.unlock();
    }
}

std::deque<EngorgioFrame*>* Controller::GetFrames(int stream_id)
{
    if (stream_context_->streams[stream_id] == nullptr) {
        return nullptr;
    }

    return &(stream_context_->streams[stream_id]->frames);
}

void Controller::DeleteFrames(int stream_id)
{
    EngorgioStream *stream = stream_context_->streams[stream_id];
    EngorgioFramePool *framepool = stream->framepool;
    EngorgioFrame *frame;

    if (stream != nullptr) {
        stream->mutex.lock();
        // std::cout << "id: " << stream_id << '\t' << "length: " << stream->frames.size() << std::endl;
        while (stream->frames.size() != 0)
        {
            frame = stream->frames.back();
            stream->frames.pop_back();
            framepool->FreeFrame(frame);
        }
        stream->mutex.unlock();
    }
}

void Controller::DeleteAnchors(int stream_id)
{
    EngorgioStream *stream = stream_context_->streams[stream_id];
    EngorgioFramePool *framepool = stream->framepool;
    EngorgioFrame *frame;

    if (stream != nullptr) {
        stream->mutex.lock();
        while (stream->anchors.size() != 0)
        {
            frame = stream->anchors.back();
            stream->anchors.pop_back();
            framepool->FreeFrame(frame);
        }
        stream->mutex.unlock();
    }
}


EngorgioStream* Controller::GetStream(int stream_id)
{
    return stream_context_->streams[stream_id];
}

bool Controller::FinishedDecode(int stream_id)
{
    bool finished = (stream_context_->streams[stream_id]->decoder == nullptr);
    return finished;
}

bool Controller::Finished(int stream_id)
{
    bool finished1 = (stream_context_->streams[stream_id]->anchors.size() == 0);
    bool finished2 = (stream_context_->streams[stream_id]->frames.size()  == 0);
    return (finished1 && finished2);
}

void Controller::LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer)
{
    anchor_engine_->LoadNeuralEnhancer(neural_enhancer);
    is_neural_enhancer_remote_ = neural_enhancer->isRemote();
}


void Controller::LoadDNNLatency(int resolution, std::string &name, double latency)
{
    anchor_engine_->LoadDNNLatency(resolution, name, latency);
}

void Controller::RunAnchorOnce()
{
    anchor_engine_->Run();
}

void Controller::RunAnchorPeriodic()
{
    anchor_engine_->LaunchPeriodic();
}

int Controller::GetTotalAnchors()
{
    return anchor_engine_->GetTotalAnchors();
}


/* deprecated
void Controller::FreeStream()
{
    std::vector<EngorgioStream*> &streams = stream_context_->streams;
    std::vector<int> &active_ids = stream_context_->active_ids;
    std::deque<int> &free_ids = stream_context_->free_ids;
    std::mutex &mutex = stream_context_->mutex;
    EngorgioStream *stream;


    auto it = active_ids.begin();
    // std::cout << "Controller: Number of active ids: " << active_ids.size() << std::endl;
    
    // std::cout << (it != active_ids.end()) << std::endl;
    while (it != active_ids.end())
    {
        stream = streams[*it];
        if (stream->decoder == nullptr && stream->frames.size() == 0)
        {
            delete stream;
            mutex.lock();
            // std::cout << "Controller: stream" << *it << " is removed"<< std::endl;
            free_ids.push_back(*it);
            it = active_ids.erase(it);
            mutex.unlock();
        }
        else
        {
            ++it;
        }
    }
}

void Controller::SetContent(int stream_id, std::string &content)
{
    std::vector<EngorgioStream*> &streams = stream_context_->streams;
    streams[stream_id]->content = content;
    // std::cout << "content: " << streams[stream_id]->content << std::endl;
}

//TODO: validate it
void Controller::FreeHandler()
{
    std::vector<int> &active_ids = stream_context_->active_ids;

    while (!exit_)
    {
        std::this_thread::sleep_for(std::chrono::seconds(interval_));
        // std::cout << "Controller: FreeHandler() is called (1)" << std::endl;
        FreeStream();
    }

    while (active_ids.size() != 0)
    {
        FreeStream();
    }
}
*/