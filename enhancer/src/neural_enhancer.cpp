#include <cassert>

#include "neural_enhancer.h"
#include "encode_engine.h"
#include "infer_engine.h"


NeuralEnhancer::NeuralEnhancer(InferEngineProfile &iprofile, bool mem_prealloc)
{
    jpeg_engine_ = nullptr;
    infer_engine_ = new InferEngine(iprofile, mem_prealloc);
    libvpx_engine_ = nullptr;
    etype_ = EnhancerType::kEngorgio;
}

NeuralEnhancer::NeuralEnhancer(InferEngineProfile &iprofile, JPEGProfile &eprofile)
{
    libvpx_engine_ = nullptr;
    jpeg_engine_ = new JPEGEncodeEngine(eprofile);
    infer_engine_ = new InferEngine(iprofile, jpeg_engine_);
    etype_ = EnhancerType::kEngorgio;
}

NeuralEnhancer::NeuralEnhancer(InferEngineProfile &iprofile, libvpxProfile &eprofile)
{
    jpeg_engine_ = nullptr;
    libvpx_engine_ = new libvpxEngine(eprofile);
    infer_engine_ = new InferEngine(iprofile, libvpx_engine_);
    etype_ = EnhancerType::kEngorgio;
}

NeuralEnhancer::~NeuralEnhancer()
{
    if (infer_engine_)
        delete infer_engine_;
    if (jpeg_engine_)
        delete jpeg_engine_;
    if (libvpx_engine_)
        delete libvpx_engine_;
}

void NeuralEnhancer::Init(int stream_id)
{
    if (libvpx_engine_)
    {
        libvpx_engine_->Init(stream_id);
    }
}

void NeuralEnhancer::Free(int stream_id)
{
    if (libvpx_engine_)
    {
        libvpx_engine_->Free(stream_id);
    }
}


// TOOD: rollback
void NeuralEnhancer::Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, bool free_model)
{
    // std::cout << "query [id:" << stream_id << "]: " << model->name_ << ", " << frames.size() << std::endl;
    assert(infer_engine_);
    infer_engine_->EnhanceAsync(stream_id, model, framepool, frames, free_model); 
}

void NeuralEnhancer::Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &deadline, bool free_model)
{
    assert(infer_engine_);
    infer_engine_->EnhanceAsync(stream_id, model, framepool, frames, deadline, free_model); 
}

bool NeuralEnhancer::Finished(int num_frames)
{
    assert(infer_engine_);
    switch(etype_)
    {
        case EnhancerType::kEngorgio:
            if (jpeg_engine_)
            {
                std::cout << "here1: " << num_frames << std::endl;
                // assert(num_frames != 0);
                if (jpeg_engine_->save_image_)
                    return jpeg_engine_->SaveFinished(num_frames);
                else   
                    return jpeg_engine_->EncodeFinished(num_frames);            
            }
            else if (libvpx_engine_)
            {
                std::cout << num_frames << std::endl;
                return libvpx_engine_->EncodeFinished(num_frames);
            }
            else
            {
                std::cout << "here2: " << num_frames << std::endl;
                return infer_engine_->Finished();
            }
    
        case EnhancerType::kPerFrame:
        case EnhancerType::kSelective:
        default:
            throw std::invalid_argument("Unsupported etype_");
    }
    
    return false;
}

bool NeuralEnhancer::LoadEngine(EngorgioModel *model)
{
    assert(infer_engine_);
    return infer_engine_->LoadEngine(model);
}

unsigned int NeuralEnhancer::GetGPUs()
{
    assert(infer_engine_);
    return infer_engine_->GetGPUs();
}