#pragma once
#include <string>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "webmdec.h"
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"

using namespace std;
namespace fs = std::filesystem;

std::string ENGORGIO_RESULT_DIR = "/workspace/research/result";
std::string ENGORGIO_VIDEO_DIR = "/workspace/research/dataset";
// std::string ENGORGIO_VIDEO_DIR = "/workspace/research/engorgio/dataset-old";

// std::vector<std::string> ENGORGIO_CONTENTS = {"chat0",  "chat1",  "couterstrike0",  "couterstrike1",
//                                               "dota20", "dota21", "fortnite0", "fortnite1",  "gta0", "gta1", "lol0", "lol1", 
//                                               "minecraft0", "minecraft1", "valorant0", "valorant1", "wow0", "wow1"};

// std::vector<std::string> ENGORGIO_CONTENTS = {"chat0", "fortnite0", "gta0", "lol0", "minecraft0", "valorant0"};
std::vector<std::string> ENGORGIO_CONTENTS = {"lol0"};


struct VpxDecInputContext
{
  struct VpxInputContext *vpx_input_ctx;
  struct WebmInputContext *webm_ctx;
};

static int dec_read_frame(struct VpxDecInputContext *input, uint8_t **buf,
                          size_t *bytes_in_buffer, size_t *buffer_size)
{
    return webm_read_frame(input->webm_ctx, buf, bytes_in_buffer);
}

struct VpxStream
{
  struct VpxDecInputContext input_;
  std::vector<pair<int, uint8_t *>> frames_;
  int stream_id_;
  std::string content_;

  VpxStream(std::string &data_dir, std::string &content, std::string &video_name, int limit = 0)
  {
    content_ = content;
    input_.vpx_input_ctx = new VpxInputContext;
    input_.webm_ctx = new WebmInputContext;
    memset(input_.webm_ctx, 0, sizeof(WebmInputContext));

    std::string video_path = data_dir + "/" + content + "/" + "video" + "/" + video_name;

    if (!fs::exists(fs::path(video_path)))
    {
        throw std::runtime_error("Video does not exist");
    }
    
    const char* fn = video_path.c_str();
    FILE *infile = fopen(fn, "rb");
    if (!infile)
    {
        throw std::runtime_error("Failed to open input file");
    }

    input_.vpx_input_ctx->file = infile;
    if (file_is_webm(input_.webm_ctx, input_.vpx_input_ctx))
    {
        input_.vpx_input_ctx->file_type = FILE_TYPE_WEBM;
    }
    else
    {
        throw std::runtime_error("Video is not a webm file");
    }

    if (input_.vpx_input_ctx->file_type == FILE_TYPE_WEBM)
    {
        if (webm_guess_framerate(input_.webm_ctx, input_.vpx_input_ctx))
        {
            throw std::runtime_error("Failed to guess framerate -- error parsing ");            
        }
    }

    int frame_avail = 1, frame_in = 0;
    size_t  buffer_size = 0;
    pair<size_t, uint8_t*> frame;
    while (frame_avail)
    {
        frame_avail = 0;
        frame.first = 0;
        frame.second = nullptr;
        if (!dec_read_frame(&input_, &frame.second, &frame.first, &buffer_size))
        {
            frame_avail = 1;
            frame_in++;
            frames_.push_back(frame); // memory free가 발생함 
        }

        if (frame_in == limit)
            break;
    }
    // std::cout << "VpxStream: " << frames_.size() << std::endl;
  }

  VpxStream(std::string &content, std::string &video_name, int limit = 0): VpxStream(ENGORGIO_VIDEO_DIR, content, video_name, limit) {}

  ~VpxStream()
  {
    if (input_.vpx_input_ctx != nullptr)
      free(input_.vpx_input_ctx);
    if (input_.webm_ctx != nullptr)
    {
      input_.webm_ctx->buffer = nullptr;
      webm_free(input_.webm_ctx);
      free(input_.webm_ctx);
    }
    for (auto frame : frames_)
    {
      free(frame.second);
    }
  }
};

VpxStream *load_stream(string &content, int resolution, int limit);


// video, image, dnn
int get_bitrate(int resolution);
std::string get_video_name(int resolution, int duration);
int get_img_size(int resolution);
std::pair<int, int> get_img_dimensions(int resolution);
std::string get_onnx_path(std::string content, std::string model, int resolution, int duration, std::string data_dir = ENGORGIO_RESULT_DIR);
std::string get_trt_path(std::string content, std::string model, int resolution, int duration, std::string data_dir = ENGORGIO_RESULT_DIR);
double get_trt_latency(std::string model, int resolution, int duration, std::string data_dir = ENGORGIO_RESULT_DIR);
void read_file(const std::string& file, void** buffer, size_t *size);

// debugging
nvinfer1::IHostMemory *load_engine(void *buf, size_t size);
void create_dummy_frames(uint8_t **buf, size_t *input_size, int num_frames, int resolution);

// pinning
std::vector<int> get_decode_threads(std::string instance, int num_gpus);
int get_anchor_thread(std::string instance);
std::vector<std::vector<int>> get_infer_threads(std::string instance, int num_gpus);
std::vector<int> get_encode_threads(std::string instance, int num_gpus);
std::vector<int> get_libvpx_threads(std::string instance, int num_gpus);
