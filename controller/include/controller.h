#pragma once 

#include <vector>
#include "control_common.h"
#include "decode_engine.h"
#include "anchor_engine.h"
#include "neural_enhancer.h"
#include "enhancer_common.h"


const int MAX_NUM_STREAMS = 1000;

// class 

class Controller {
private:
    // anchor scheduler
    EngorgioStreamContext *stream_context_;
    DecodeEngine *decode_engine_;
    AnchorEngine *anchor_engine_;
    bool is_neural_enhancer_remote_ {false};
public:
    Controller(DecodeEngineProfile &dprofile, AnchorEngineProfile &aprofile);
    ~Controller();

    // process (online)
    int Init(int gop, std::string &content, EngorgioModel* model = nullptr);
    void Process(int stream_id, uint8_t *buf, int len);
    void Free(int stream_id);
    void Free();
    bool FinishedDecode(int stream_id);
    bool Finished(int stream_id);

    // load (offline)
    void LoadNeuralEnhancer(NeuralEnhancer *neural_enhancer);
    void LoadDNNLatency(int resolution, std::string &name, double latency);

    // deprecated (for debugging)
    void RunAnchorOnce();
    void RunAnchorPeriodic();
    std::deque<EngorgioFrame*>* GetFrames(int stream_id); // for debugging
    void DeleteFrames(int stream_id); // for debugging
    void DeleteAnchors(int stream_id); // for debugging
    EngorgioStream* GetStream(int stream_id);
    int GetTotalAnchors();
};