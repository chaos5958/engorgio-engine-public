#pragma once 
#include "encode_engine.h"
#include "infer_engine.h"
#include "libvpx_engine.h"
#include "enhancer_common.h"

enum class EnhancerType : int
{
    kEngorgio = 0,
    kPerFrame = 1,
    kSelective = 2,
};

class NeuralEnhancerServer;

class NeuralEnhancer
{
protected:
    EnhancerType etype_;
    InferEngine *infer_engine_;
    JPEGEncodeEngine *jpeg_engine_;
    libvpxEngine *libvpx_engine_;

    friend class NeuralEnhancerServer;

public:
    NeuralEnhancer(InferEngineProfile &iprofile, bool mem_prealloc = true);
    NeuralEnhancer(InferEngineProfile &iprofile, JPEGProfile &eprofile);
    NeuralEnhancer(InferEngineProfile &iprofile, libvpxProfile &eprofile);
    virtual ~NeuralEnhancer();

    void Init(int stream_id);
    virtual void Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, bool free_model = true);
    virtual void Process(int stream_id, EngorgioModel *model, EngorgioFramePool *framepool, std::vector<EngorgioFrame*> &frames, std::chrono::system_clock::time_point &dealine, bool free_model = true);
    void Free(int stream_id);

    virtual bool LoadEngine(EngorgioModel *model);
    // bool Finished();
    virtual bool Finished(int num_frames = 0);
    virtual unsigned int GetGPUs();
    virtual bool isRemote() const { return false; }
};