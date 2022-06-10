#pragma once

#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct VpxDecInputContext
{
  struct VpxInputContext *vpx_input_ctx;
  struct WebmInputContext *webm_ctx;
};

struct VpxStream
{
  struct VpxDecInputContext input;
  std::vector<pair<int, uint8_t *>> frames;
  int stream_id;
  std::string content;

  VpxStream()
  {
    input.vpx_input_ctx = new VpxInputContext;
    input.webm_ctx = new WebmInputContext;

    memset(input.webm_ctx, 0, sizeof(WebmInputContext));
  }

  VpxStream(std::string &content_)
  {
    content = content_;
    input.vpx_input_ctx = new VpxInputContext;
    input.webm_ctx = new WebmInputContext;

    memset(input.webm_ctx, 0, sizeof(WebmInputContext));
  }

  ~VpxStream()
  {
    if (input.vpx_input_ctx != nullptr)
      free(input.vpx_input_ctx);
    if (input.webm_ctx != nullptr)
    {
      input.webm_ctx->buffer = nullptr;
      webm_free(input.webm_ctx);
      free(input.webm_ctx);
    }
    for (auto frame : frames)
    {
      free(frame.second);
    }
  }
};

VpxStream *load_stream(string &content, int resolution, int limit);

/* deprecated */
extern std::string video_dir;
extern std::string log_dir;
int load_video(string &video_file, vector<pair<int, uint8_t *>> &video_frames);
