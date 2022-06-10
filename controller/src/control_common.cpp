#include "control_common.h"
#include "enhancer_common.h"

void free_frame(EngorgioFrame *frame)
{
    if (frame != nullptr)
    {
        if (frame->rgb_buffer != nullptr)
            free(frame->rgb_buffer);
    }
    free(frame);
}

EngorgioStream::EngorgioStream(int gop_, std::string &content_, EngorgioModel *model_)
{
    avg_size = 0;
    content = content_;
    framepool = new EngorgioFramePool(INIT_NUM_FRAMES, 1920, 1080);
    decoder = nullptr;
    prev_total_residual = 0;
    gop = gop_;
    model = model_;
    anchors.reserve(INIT_NUM_FRAMES);
}
EngorgioStream::~EngorgioStream()
{
    delete framepool;
    delete model;

    for (auto log : dlatency_logs)
        delete log;
    for (auto log : aaindex_logs)
        delete log;
    for (auto log : afindex_logs)
        delete log;
}
