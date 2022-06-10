#include <fstream>
#include <turbojpeg.h>
#include "control_common.h"
#include "libvpx_engine.h"
#include "vpxenc_api.h"
#include "./vpx/vpx_codec.h"

static size_t read_file(const std::string& file, void** buffer)
{
    size_t size;

    std::ifstream stream(file, std::ifstream::ate | std::ifstream::binary);
    if (!stream.is_open())
    {
        return 0;
    }

    size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    *buffer = (unsigned char*)malloc(size);
    if (*buffer == nullptr)
        return 0;

    stream.read((char*)*buffer, size);
    return size;
}

EngorgioFrame* LoadRGBFrame(int video_index, int width, int height, std::string image_path)
{
    EngorgioFrame *frame = new EngorgioFrame();
    frame->current_video_frame = video_index;
    frame->current_super_frame = 0;
    frame->width = width;
    frame->height = height;
    read_file(image_path, (void**)&frame->rgb_buffer);
    // std::cout << frame->width << "," << frame->height << std::endl;
    return frame;
}

int main()
{
    // profile
    int bitrate = 30000;
    int num_threads = 16;
    std::vector<int> thread_indexes = {};
    for (int i = 0; i <  16; i ++)
        thread_indexes.push_back(i);
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    libvpxProfile profile = libvpxProfile(bitrate, num_threads, true, false, global_start, thread_indexes);
    
    // load rgb images
    int height = 2160;
    int width = 3840;
    EngorgioFrame* frame;
    std::string image_dir = "/workspace/research/frames";
    std::string image_path;
    char buffer[256];
    std::vector<EngorgioFrame*> frames;
    
    int num_frames = 600;
    for (int j = 1; j <= num_frames; j++)
    {
        sprintf(buffer, "%04d",  j);
        image_path = image_dir + "/" + std::string(buffer) + ".raw";
        frame = LoadRGBFrame(j, width, height, image_path);
        frames.push_back(frame);
    }   

    // run test (throughput)
    int stream_id = 0;

    libvpxEngine *engine = new libvpxEngine(profile);
    engine->Init(stream_id);
    engine->Encode(stream_id, frames);
    while(!engine->EncodeFinished(num_frames))
    {}
    engine->Free(stream_id);
    delete engine;

    for (auto &frame : frames)
        delete frame;
}