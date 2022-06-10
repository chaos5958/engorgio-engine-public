#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "webmdec.h"
#include "DecodeEngine.h"

using namespace std;

std::string video_dir = "/workspace/research/NeuralEnhancer-data/product_review/video";
std::string log_dir = "/workspace/research/NeuralEnhancer-data/test/Engine";

struct VpxDecInputContext {
  struct VpxInputContext *vpx_input_ctx;
  struct WebmInputContext *webm_ctx;
};

static int dec_read_frame(struct VpxDecInputContext *input, uint8_t **buf,
                          size_t *bytes_in_buffer, size_t *buffer_size)
{
    return webm_read_frame(input->webm_ctx, buf, bytes_in_buffer);
}

struct test_config
{
    int delay_in_ms;
    int resolution;
    size_t limit;

    map<int, vector<pair<int, uint8_t*>>> *video_frames_map;
    DecodeEngine *dengine;
};

int load_video(const string &video_file, vector<pair<int, uint8_t*>> &video_frames)
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
    uint8_t *buf = NULL;
    size_t bytes_in_buffer = 0, buffer_size = 0;
    while (frame_avail)
    {
        frame_avail = 0;
        
        pair<size_t, uint8_t*> video_frame;
        if (!dec_read_frame(&input, &video_frame.second, &video_frame.first, &buffer_size))
        {
            frame_avail = 1;
            frame_in++;
            video_frames.push_back(video_frame); // memory free가 발생함 
            // cout << "Bytes: " << bytes_in_buffer << endl;
            // cout << "Buf: " << buf << endl;
        }
    }

    return 0;
}

// run in multi-threads
// Q. how to check decoding is finished (before releasing a thread)?
void test_video(const test_config *test_config, int stream_id)
{
    DecodeEngine *dengine = test_config->dengine;
    vector<pair<int, uint8_t*>> &video_frames = test_config->video_frames_map->at(test_config->resolution);
    
    dengine->DecoderInit(stream_id);
    int max_idx;
    if (test_config->limit) 
    {
        max_idx = min(video_frames.size(), test_config->limit);
    }
    else{
        max_idx = video_frames.size();
    }
    for (int i = 0; i < max_idx; i++)
    {
        dengine->DecoderDecode(stream_id, video_frames[i].second, video_frames[i].first);
    }
    dengine->DecoderDestroy(stream_id);    
}


void test_videos(vector<test_config> &test_configs)
{
    //run multi-thread
    DecodeEngine *dengine = new DecodeEngine(EngineType::kBaseline, 1, log_dir);
    dengine->Build();

    vector<thread> threads;
    test_config *config;
    for (int i = 0; i < test_configs.size(); i++)
    {
        config = &test_configs[i];
        config->dengine = dengine;
        threads.emplace_back(std::thread([config, i](){test_video(config, i);}));
    }

    for (auto &thread: threads)
    {
        thread.join();
    }
    dengine->Destroy();
}
  

void test_single_stream(map<int, vector<pair<int, uint8_t*>>> *video_frames_map)
{
    //build decode engine
    vector<test_config> configs;
    test_config config {0, 1080, 100, video_frames_map, nullptr};
    configs.push_back(config);
    test_videos(configs);
}

void test_multi_stream(map<int, vector<pair<int, uint8_t*>>> *video_frames_map)
{
    //build decode engine
    vector<test_config> configs;
    test_config config {0, 1080, 0, video_frames_map, nullptr};
    configs.push_back(config);
    configs.push_back(config);
    configs.push_back(config);
    configs.push_back(config);
    configs.push_back(config);
    test_videos(configs);
}

int main()
{
    map<int, string> video_files_map;
    map<int, vector<pair<int, uint8_t*>>> video_frames_map;
    // video_files_map[240] = video_dir + "/240p_512kbps_s0_d300.webm";
    // video_files_map[360] = video_dir + "/360p_1024kbps_s0_d300.webm";
    // video_files_map[480] = video_dir + "/480p_1600kbps_s0_d300.webm";
    // video_files_map[720] = video_dir + "/720p_2640kbps_s0_d300.webm";
    video_files_map[1080] = video_dir + "/1080p_4400kbps_s0_d300.webm";

    for (auto const& x: video_files_map)
    {
        if (load_video(x.second, video_frames_map[x.first]))
        {
            cerr << "load_video() failed" << endl;
            return -1;
        }
    }

    test_single_stream(&video_frames_map);
    test_multi_stream(&video_frames_map);

    return 0;
}