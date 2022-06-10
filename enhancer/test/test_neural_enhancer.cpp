#include <fstream>
#include <turbojpeg.h>
#include "control_common.h"
#include "neural_enhancer.h"
#include "encode_engine.h"
#include "infer_engine.h"
#include "tool_common.h"

const int NUM_DUMMY_ITERS = 10;

void test_throughput(InferEngineProfile &iprofile, JPEGProfile &eprofile, int stream_id, 
                            std::vector<EngorgioModel*> &onnx_model_per_request, EngorgioModel *trt_model,
                            std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
{
    NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    // NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile);
    neural_enhancer->LoadEngine(trt_model);
    
    // dummy
    int num_frames = NUM_DUMMY_ITERS * frames_per_request[0].size();
    for (int i = 0; i < NUM_DUMMY_ITERS; i++)
    {
        neural_enhancer->Process(0, onnx_model_per_request[i], nullptr, frames_per_request[i]);
    }
    while (!neural_enhancer->Finished(num_frames))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // actual
    num_frames += num_repeats * frames_per_request[0].size();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_repeats; i++)
    {
        neural_enhancer->Process(0, onnx_model_per_request[i+NUM_DUMMY_ITERS], nullptr, frames_per_request[i+NUM_DUMMY_ITERS]);
    }
    while (!neural_enhancer->Finished(num_frames))
    {
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (End-to-end): " << elapsed.count() * 1000 << "ms" << std::endl;

    delete neural_enhancer;
}

void test_quality(InferEngineProfile &iprofile, JPEGProfile &eprofile, int stream_id, 
                           EngorgioModel* onnx_model, EngorgioModel *trt_model,
                           std::vector<EngorgioFrame*> &frames)
{
    NeuralEnhancer *neural_enhancer = new NeuralEnhancer(iprofile, eprofile);
    neural_enhancer->LoadEngine(trt_model);
    
    // actual
    int num_frames = frames.size();
    auto start = std::chrono::high_resolution_clock::now();
    neural_enhancer->Process(0, onnx_model, nullptr, frames);
    while (!neural_enhancer->Finished(num_frames))
    {
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (End-to-end): " << elapsed.count() * 1000 << "ms" << std::endl;

    delete neural_enhancer;
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

// TODO1: create dataset with lol0
// TODO2: update code to lolo0 
int main()
{
    // profile
    int num_repeats = 100;
    int qp = 80;
    int num_workers = 8;

    // images
    int num_frames = 6;
    int height = 720;
    int width = 1280;
    int duration = 600;
    int stream_id = 0;
    std::string image_dir = "/workspace/research/engorgio/result/key";
    std::string image_path;
    char buffer[256];

    // dnns
    int scale = 3;
    std::string content = "lol0";
    std::string model_name = "EDSR_B8_F32_S3";
    std::string onnx_path = get_onnx_path(content, model_name, height, duration);
    std::string trt_path = get_trt_path(content, model_name, height, duration);
    EngorgioFrame* frame;
    EngorgioModel *trt_model, *onnx_model;

    /****************** throughput ********************/
    // // profiles
    std::chrono::system_clock::time_point global_start = std::chrono::system_clock::now();
    // std::vector<int> thread_indexes = {2, 4};
    JPEGProfile eprofile = JPEGProfile(qp, num_workers, true, false, global_start);
    InferEngineProfile iprofile = InferEngineProfile(InferEngineType::kEngorgio, true, 1);
    
    // load rgb images
    std::vector<std::vector<EngorgioFrame*>> frames_per_request;
    assert(num_frames < 10);
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        std::vector<EngorgioFrame*> frames;
        for (int j = 1; j <= num_frames; j++)
        {
            sprintf(buffer, "%04d",  j);
            image_path = image_dir + "/" + std::string(buffer) + ".rgb";
            frame = LoadRGBFrame(j, width, height, image_path);
            frames.push_back(frame);
        }   
        frames_per_request.push_back(frames);
    }

    // load dnns
    std::vector<EngorgioModel *> onnx_model_per_request;
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        onnx_model = new EngorgioModel(onnx_path, model_name, scale);
        onnx_model_per_request.push_back(onnx_model);
    }
    trt_model = new EngorgioModel(trt_path, model_name, scale);
    
    // run test (throughput)
    test_throughput(iprofile, eprofile, stream_id, onnx_model_per_request, trt_model,
                    frames_per_request, num_repeats);
    
    /****************** quality ********************/
    // load a dnn
    // onnx_model = new EngorgioModel(onnx_path, model_name, scale);
    // trt_model = new EngorgioModel(trt_path, model_name, scale);

    // // load frames
    // std::vector<EngorgioFrame*> frames_q;
    // for (int j = 1; j <= num_frames; j++)
    // {
    //     sprintf(buffer, "%04d",  j);
    //     image_path = image_dir + "/" + std::string(buffer) + ".rgb";
    //     frame = LoadRGBFrame(j, width, height, image_path);
    //     frames_q.push_back(frame);
    // }   
    // // load dnns
    // InferEngineProfile iprofile_q = InferEngineProfile(InferEngineType::kEngorgio, false, true);
    // JPEGProfile eprofile_q = JPEGProfile(qp, num_workers, std::filesystem::path(""));
    // test_quality(iprofile_q, eprofile_q, stream_id, onnx_model, trt_model, frames_q);
}