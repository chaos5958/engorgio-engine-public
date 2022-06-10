#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include "webmdec.h"
#include "controller.h"
#include "control_common.h"
#include "test_common.h"

namespace fs = std::filesystem;

std::string video_dir = "/workspace/research/NeuralEnhancer-data/product_review/video";
std::string log_dir = "/workspace/research/NeuralEnhancer-data/test/Engine";

static int dec_read_frame(struct VpxDecInputContext *input, uint8_t **buf,
                          size_t *bytes_in_buffer, size_t *buffer_size)
{
    return webm_read_frame(input->webm_ctx, buf, bytes_in_buffer);
}

std::string base_dir = "/workspace/research/engorgio-dataset";

static std::string video_name(int resolution)
{
    switch(resolution)
    {
        case 360:
            return "360p_1536kbps_s0_d60.webm";
        case 720:
            return "720p_3960kbps_s0_d60.webm";
        default:
            throw invalid_argument("Invalid resolution");
    }
}


VpxStream* load_stream(string &content, int resolution, int limit)
{
    std::string video_path = base_dir + "/" + content + "/" + "video" + "/" + video_name(resolution);

    if (!fs::exists(fs::path(video_path)))
    {
        throw runtime_error("Video does not exist");
    }
    
    VpxStream *stream = new VpxStream(content);
    const char* fn = video_path.c_str();
    FILE *infile = fopen(fn, "rb");
    if (!infile)
    {
        fprintf(stderr, "Failed to open input file '%s'", strcmp(fn, "-") ? fn : "stdin");
        return nullptr; 
    }
    stream->input.vpx_input_ctx->file = infile;
    if (file_is_webm(stream->input.webm_ctx, stream->input.vpx_input_ctx))
    {
        stream->input.vpx_input_ctx->file_type = FILE_TYPE_WEBM;
    }
    else
    {
        cerr << "Video is not a webm file" << endl;
        delete stream;
        return nullptr;
    }

    if (stream->input.vpx_input_ctx->file_type == FILE_TYPE_WEBM)
    {
        if (webm_guess_framerate(stream->input.webm_ctx, stream->input.vpx_input_ctx))
        {
            fprintf(stderr,
                    "Failed to guess framerate -- error parsing "
                    "webm file?\n");
            delete stream;
            return nullptr;
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
        if (!dec_read_frame(&stream->input, &frame.second, &frame.first, &buffer_size))
        {
            frame_avail = 1;
            frame_in++;
            stream->frames.push_back(frame); // memory free가 발생함 
            // cout << "Bytes: " << bytes_in_buffer << endl;
            // cout << "Buf: " << buf << endl;
        }

        if (frame_in == limit)
            break;
    }

    return stream;
}

// TODO: when to destroy webm context and buffers 
// TODO: free buffers, free webm context (set webm->buffers = nullptr;)


int load_video(string &video_file, vector<pair<int, uint8_t*>> &video_frames)
{
    // cout << "video file: " << video_file << endl;
    struct VpxDecInputContext input = {NULL, NULL};
    struct VpxInputContext vpx_input_ctx;
    struct WebmInputContext webm_ctx;
    memset(&(webm_ctx), 0, sizeof(webm_ctx));

    const char* fn = video_file.c_str();
    FILE *infile = fopen(fn, "rb");
    if (!infile)
    {
        fprintf(stderr, "Failed to open input file '%s'", strcmp(fn, "-") ? fn : "stdin");
        return -1; 
    }

    input.webm_ctx = &webm_ctx;
    input.vpx_input_ctx = &vpx_input_ctx;
    input.vpx_input_ctx->file = infile;
    if (file_is_webm(input.webm_ctx, input.vpx_input_ctx))
    {
        input.vpx_input_ctx->file_type = FILE_TYPE_WEBM;
    }
    else
    {
        cerr << "Video is not a webm file" << endl;
        return -1;
    }

    if (vpx_input_ctx.file_type == FILE_TYPE_WEBM)
    {
        if (webm_guess_framerate(input.webm_ctx, input.vpx_input_ctx))
        {
            fprintf(stderr,
                    "Failed to guess framerate -- error parsing "
                    "webm file?\n");
            return -1;
        }
    }

    int frame_avail = 1, frame_in = 0;
    size_t buffer_size = 0;
    int i = 0;
    while (frame_avail)
    {
        frame_avail = 0;
        
        pair<size_t, uint8_t*> video_frame;
        // std::cout << "buffer_size (before): " << video_frame.first << std::endl;
        if (!dec_read_frame(&input, &video_frame.second, &video_frame.first, &buffer_size))
        {
            frame_avail = 1;
            frame_in++;
            video_frames.push_back(video_frame); // memory free가 발생함 
            // cout << "Bytes: " << bytes_in_buffer << endl;
            // cout << "Buf: " << buf << endl;
            // std::cout << "address: " << reinterpret_cast<void *>(video_frame.second) << std::endl;
            // std::cout << "buffer_size: " << video_frame.first << std::endl;
        }
        i += 1;
        if (i == 10)
            break;
    }

    cout << "Total number of frames: " << video_frames.size() << endl;

    return 0;
}