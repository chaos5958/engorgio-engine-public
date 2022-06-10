#include <fstream>
#include <turbojpeg.h>
#include "control_common.h"
#include "encode_engine.h"

const int NUM_DUMMY_ITERS = 1;

void test_throughput(JPEGProfile &profile, int stream_id, std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
{
    JPEGEncodeEngine *encode_engine = new JPEGEncodeEngine(profile);

    // TODO: dummy
    for (int i = 0; i < NUM_DUMMY_ITERS; i++)
    {
        encode_engine->Encode(stream_id, frames_per_request[i]);
    }
    while(!encode_engine->EncodeFinished(NUM_DUMMY_ITERS * frames_per_request[0].size()))
    {}

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_repeats; i++)
    {
        encode_engine->Encode(stream_id, frames_per_request[i + NUM_DUMMY_ITERS]);
    }
    while(!encode_engine->EncodeFinished((num_repeats + NUM_DUMMY_ITERS) * frames_per_request[0].size()))
    {}
    while(!encode_engine->SaveFinished((num_repeats + NUM_DUMMY_ITERS) * frames_per_request[0].size()))
    {}

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
    std::cout << "Throughput (Encode): " << (num_repeats * frames_per_request[0].size()) / (elapsed.count()) << "fps" << std::endl;

    delete encode_engine;
}

void test_quality(JPEGProfile &profile, int stream_id, std::vector<EngorgioFrame*> &frames)
{
    JPEGEncodeEngine *encode_engine = new JPEGEncodeEngine(profile);
    encode_engine->Encode(stream_id, frames);

    while(!encode_engine->SaveFinished(frames.size()))
    {}

    delete encode_engine;
}

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
    int qp = 80;
    int num_workers = 1;
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    std::vector<int> thread_indexes = {2, 4};
    JPEGProfile profile = JPEGProfile(qp, num_workers, true, true, global_start, thread_indexes);

    // load rgb images
    int height = 720;
    int width = 1280;
    int stream_id = 0;
    std::vector<std::vector<EngorgioFrame*>> frames_per_request;
    EngorgioFrame* frame;
    std::string image_dir = "/workspace/research/engorgio/result/key";
    std::string image_path;
    char buffer[256];

    // run test (throughput)
    int num_repeats = 100;
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        std::vector<EngorgioFrame*> frames;
        for (int j = 1; j <= 15; j++)
        {
            sprintf(buffer, "%04d",  j);
            image_path = image_dir + "/" + std::string(buffer) + ".rgb";
            frame = LoadRGBFrame(j, width, height, image_path);
            frames.push_back(frame);
        }   
        frames_per_request.push_back(frames);
    }
    test_throughput(profile, stream_id, frames_per_request, num_repeats);

    // run test (quality)
    // EncodeProfile profile1 = EncodeProfile(qp, num_workers, std::filesystem::path(""));
    //     std::vector<EngorgioFrame*> frames;
    //     for (int i = 1; i <= 15; i++)
    //     {
    //         sprintf(buffer, "%04d", i);
    //         image_path = image_dir + "/" + std::string(buffer) + ".rgb";
    //         frame = LoadRGBFrame(i, width, height, image_path);
    //         frames.push_back(frame);
    //     }   
   // test_quality(profile1, stream_id, frames_per_request);
}