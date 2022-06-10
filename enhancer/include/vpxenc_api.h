/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPXENC_H_
#define VPX_VPXENC_H_

#include "./vpx_config.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if CONFIG_LIBYUV
#include "third_party/libyuv/include/libyuv/scale.h"
#endif

#include "vpx/vpx_encoder.h"
#if CONFIG_DECODERS
#include "vpx/vpx_decoder.h"
#endif

#include "./args.h"
#include "./ivfenc.h"
#include "./tools_common.h"

#if CONFIG_VP8_ENCODER || CONFIG_VP9_ENCODER
#include "vpx/vp8cx.h"
#endif
#if CONFIG_VP8_DECODER || CONFIG_VP9_DECODER
#include "vpx/vp8dx.h"
#endif

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem_ops.h"
#include "vpx_ports/vpx_timer.h"
#include "./rate_hist.h"
#include "./vpxstats.h"
#include "./warnings.h"
#if CONFIG_WEBM_IO
#include "./webmenc.h"
#endif
#include "./y4minput.h"

#ifdef __cplusplus
extern "C" {
#endif

enum TestDecodeFatality {
  TEST_DECODE_OFF,
  TEST_DECODE_FATAL,
  TEST_DECODE_WARN,
};

typedef enum {
  I420,  // 4:2:0 8+ bit-depth
  I422,  // 4:2:2 8+ bit-depth
  I444,  // 4:4:4 8+ bit-depth
  I440,  // 4:4:0 8+ bit-depth
  YV12,  // 4:2:0 with uv flipped, only 8-bit depth
  NV12,  // 4:2:0 with uv interleaved
} ColorInputType;

struct VpxInterface;

/* Configuration elements common to all streams. */
struct VpxEncoderConfig {
  const struct VpxInterface *codec;
  int passes;
  int pass;
  int usage;
  int deadline;
  ColorInputType color_type;
  int quiet;
  int verbose;
  int limit;
  int skip_frames;
  int show_psnr;
  enum TestDecodeFatality test_decode;
  int have_framerate;
  struct vpx_rational framerate;
  int out_part;
  int debug;
  int show_q_hist_buckets;
  int show_rate_hist_buckets;
  int disable_warnings;
  int disable_warning_prompt;
  int experimental_bitstream;
};


static const int vp8_arg_ctrl_map[] = { VP8E_SET_CPUUSED,
                                        VP8E_SET_ENABLEAUTOALTREF,
                                        VP8E_SET_NOISE_SENSITIVITY,
                                        VP8E_SET_SHARPNESS,
                                        VP8E_SET_STATIC_THRESHOLD,
                                        VP8E_SET_TOKEN_PARTITIONS,
                                        VP8E_SET_ARNR_MAXFRAMES,
                                        VP8E_SET_ARNR_STRENGTH,
                                        VP8E_SET_ARNR_TYPE,
                                        VP8E_SET_TUNING,
                                        VP8E_SET_CQ_LEVEL,
                                        VP8E_SET_MAX_INTRA_BITRATE_PCT,
                                        VP8E_SET_GF_CBR_BOOST_PCT,
                                        VP8E_SET_SCREEN_CONTENT_MODE,
                                        0 };

static const int vp9_arg_ctrl_map[] = { VP8E_SET_CPUUSED,
                                        VP8E_SET_ENABLEAUTOALTREF,
                                        VP8E_SET_SHARPNESS,
                                        VP8E_SET_STATIC_THRESHOLD,
                                        VP9E_SET_TILE_COLUMNS,
                                        VP9E_SET_TILE_ROWS,
                                        VP9E_SET_TPL,
                                        VP8E_SET_ARNR_MAXFRAMES,
                                        VP8E_SET_ARNR_STRENGTH,
                                        VP8E_SET_ARNR_TYPE,
                                        VP8E_SET_TUNING,
                                        VP8E_SET_CQ_LEVEL,
                                        VP8E_SET_MAX_INTRA_BITRATE_PCT,
                                        VP9E_SET_MAX_INTER_BITRATE_PCT,
                                        VP9E_SET_GF_CBR_BOOST_PCT,
                                        VP9E_SET_LOSSLESS,
                                        VP9E_SET_FRAME_PARALLEL_DECODING,
                                        VP9E_SET_AQ_MODE,
                                        VP9E_SET_ALT_REF_AQ,
                                        VP9E_SET_FRAME_PERIODIC_BOOST,
                                        VP9E_SET_NOISE_SENSITIVITY,
                                        VP9E_SET_TUNE_CONTENT,
                                        VP9E_SET_COLOR_SPACE,
                                        VP9E_SET_MIN_GF_INTERVAL,
                                        VP9E_SET_MAX_GF_INTERVAL,
                                        VP9E_SET_TARGET_LEVEL,
                                        VP9E_SET_ROW_MT,
                                        VP9E_SET_DISABLE_LOOPFILTER,
                                        0 };

#define NELEMENTS(x) (sizeof(x) / sizeof(x[0]))
#if CONFIG_VP9_ENCODER
#define ARG_CTRL_CNT_MAX NELEMENTS(vp9_arg_ctrl_map)
#else
#define ARG_CTRL_CNT_MAX NELEMENTS(vp8_arg_ctrl_map)
#endif


struct stream_config {
  struct vpx_codec_enc_cfg cfg;
  const char *out_fn;
  const char *stats_fn;
  stereo_format_t stereo_fmt;
  int arg_ctrls[ARG_CTRL_CNT_MAX][2];
  int arg_ctrl_cnt;
  int write_webm;
};

struct stream_state {
  int index;
  struct stream_state *next;
  struct stream_config config;
  FILE *file;
  struct rate_hist *rate_hist;
  struct WebmOutputContext webm_ctx;
  uint64_t psnr_sse_total;
  uint64_t psnr_samples_total;
  double psnr_totals[4];
  int psnr_count;
  int counts[64];
  vpx_codec_ctx_t encoder;
  unsigned int frames_out;
  uint64_t cx_time;
  size_t nbytes;
  stats_io_t stats;
  struct vpx_image *img;
  vpx_codec_ctx_t decoder;
  int mismatch_seen;
};

struct stream_state *new_stream(struct VpxEncoderConfig *global,
                                       struct stream_state *prev);


void parse_stream_params(struct VpxEncoderConfig *global,
                               struct stream_state *stream);                

void set_stream_dimensions(struct stream_state *stream, unsigned int w,
                                  unsigned int h);
void validate_stream_config(const struct stream_state *stream,
                                   const struct VpxEncoderConfig *global);                                                      
void show_stream_config(struct stream_state *stream,
                               struct VpxEncoderConfig *global,
                               struct VpxInputContext *input); 
void setup_pass(struct stream_state *stream,
                       struct VpxEncoderConfig *global, int pass);   

void open_output_file(struct stream_state *stream,
                             struct VpxEncoderConfig *global,
                             const struct VpxRational *pixel_aspect_ratio);
void initialize_encoder(struct stream_state *stream,
                               struct VpxEncoderConfig *global);     
float usec_to_fps(uint64_t usec, unsigned int frames);                  
void print_time(const char *label, int64_t etl);     
void encode_frame(struct stream_state *stream,
                         struct VpxEncoderConfig *global, struct vpx_image *img,
                         unsigned int frames_in);
void close_output_file(struct stream_state *stream,
                              unsigned int fourcc);                                                                                                        


void get_cx_data(struct stream_state *stream,
                        struct VpxEncoderConfig *global, int *got_data);

void set_arg_ctrl(struct stream_config *config, const int *ctrl_args_map, int ctrl, int value);                        

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPXENC_H_