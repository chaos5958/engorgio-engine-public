/*****************************************************************************/
// File: simple_example_c.cpp [scope = APPS/SIMPLE_EXAMPLE]
// Version: Kakadu, V8.2.1
// Author: David Taubman
// Last Revised: 17 September, 2021
/*****************************************************************************/
// Copyright 2001, David Taubman.  The copyright to this file is owned by
// Kakadu R&D Pty Ltd and is licensed through Kakadu Software Pty Ltd.
// Neither this copyright statement, nor the licensing details below
// may be removed from this file or dissociated from its contents.
/*****************************************************************************/
// Licensee: Hyunho Yeo - Personal Use
// License number: 02079
// The licensee has been granted a (non-HT) INDIVIDUAL NON-COMMERCIAL license
// to the contents of this source file.  A brief summary of this license appears
// below.  This summary is not to be relied upon in preference to the full
// text of the license agreement, accepted at purchase of the license.
// 1. The Licensee has the right to install and use the Kakadu software and
//    to develop Applications for the Licensee's own individual use.
// 2. The Licensee has the right to Deploy Applications built using the
//    Kakadu software to Third Parties, so long as such Deployment does not
//    result in any direct or indirect financial return to the Licensee or
//    any other Third Party, which further supplies or otherwise uses such
//    Applications, and provided Kakadu's HT block encoder/decoder
//    implementation remains disabled, unless explicit permission has been
//    granted to the Licensee to deploy Applications with HT enabled.
// 3. The Licensee has the right to distribute Reusable Code (including
//    source code and dynamically or statically linked libraries) to a Third
//    Party, provided the Third Party possesses a suitable license to use the
//    Kakadu software, and provided such distribution does not result in any
//    direct or indirect financial return to the Licensee.
// 4. The Licensee has the right to enable Kakadu's HT block encoder/decoder
//    implementation for evaluation and personal development purposes, but
//    not for deployed Applications.
/******************************************************************************
Description:
   Simple example showing compression with an intermediate buffer used to
store the image samples.  This is, of course, a great waste of memory and
the more sophisticated "kdu_compress" and "kdu_buffered_compress" applications
do no such thing.  However, developers may find this type of introductory
example helpful as a first exercise in understanding how to use Kakadu.  The
vast majority of Kakadu's interesting features are not utilized by this
example, but it may be enough to satisfy the initial needs of some developers
who are working with moderately large images, already in an internal buffer.
For a more sophisticated compression application, which is still quite
readily comprehended, see the "kdu_buffered_compress" application.
   The emphasis here is on showing a minimal number of instructions required
to open a codestream object, create a processing engine and push an image into
the engine, producing a given number of automatically selected rate-distortion
optimized quality layers for a quality-progressive representation.
   You may find it interesting to open the compressed codestream using
the "kdu_show" utility, examine the properties (file:properties menu item),
and investigate the quality progression associated with the default set of
quality layers created here (use the "<" and ">" keys to navigate through
the quality layers interactively).
******************************************************************************/

// System includes
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
// load functions
// #include "kdu_functions.h"

#include <chrono>
#include <dirent.h>
#include <errno.h>
#include "cxxopts.hpp"
#include "nlohmann/json.hpp"

// using namespace kdu_supp; // Also includes the `kdu_core' namespace

#include <turbojpeg.h>
#include "control_common.h"
#include "encode_engine.h"
#include "tool_common.h"

const int NUM_DUMMY_ITERS = 1;
namespace fs = std::filesystem;
using json = nlohmann::json;


std::vector<double> test_throughput(JPEGProfile &profile, int stream_id, std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
    std::cout << "Throughput (Encode): " << (num_repeats * frames_per_request[0].size()) / (elapsed.count()) << "fps" << std::endl;

    delete encode_engine;

    std::vector<double> results;
    results.push_back(elapsed.count());
    results.push_back(num_repeats * frames_per_request[0].size()); 
    results.push_back((num_repeats * frames_per_request[0].size()) / (elapsed.count()));
    return results;
}

std::vector<double> test_throughput_kakadu(JPEGProfile &profile, int stream_id, std::vector<std::vector<EngorgioFrame*>> &frames_per_request, int num_repeats)
{
    JPEGEncodeEngine *encode_engine = new JPEGEncodeEngine(profile);

    // TODO: dummy
    for (int i = 0; i < NUM_DUMMY_ITERS; i++)
    {
        encode_engine->Encode_kakadu(stream_id, frames_per_request[i]);
    }
    while(!encode_engine->EncodeFinished(NUM_DUMMY_ITERS * frames_per_request[0].size()))
    {}

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_repeats; i++)
    {
        encode_engine->Encode_kakadu(stream_id, frames_per_request[i + NUM_DUMMY_ITERS]);
    }
    while(!encode_engine->EncodeFinished((num_repeats + NUM_DUMMY_ITERS) * frames_per_request[0].size()))
    {}

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Latency (Encode): " << elapsed.count() * 1000 << "ms" << std::endl;
    std::cout << "Throughput (Encode): " << (num_repeats * frames_per_request[0].size()) / (elapsed.count()) << "fps" << std::endl;

    
    delete encode_engine;

    std::vector<double> results;
    results.push_back(elapsed.count());
    results.push_back(num_repeats * frames_per_request[0].size()); 
    results.push_back((num_repeats * frames_per_request[0].size()) / (elapsed.count()));
    return results;
}

void test_quality(JPEGProfile &profile, int stream_id, std::vector<EngorgioFrame*> &frames)
{
    JPEGEncodeEngine *encode_engine = new JPEGEncodeEngine(profile);
    encode_engine->Encode(stream_id, frames);

    while(!encode_engine->SaveFinished(frames.size()))
    {}

    delete encode_engine;
}


void test_quality_kakadu(JPEGProfile &profile, int stream_id, std::vector<EngorgioFrame*> &frames)
{
    JPEGEncodeEngine *encode_engine = new JPEGEncodeEngine(profile);
    encode_engine->Encode_kakadu(stream_id, frames);

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

int main(int argc, char** argv)
{
    cxxopts::Options options("InferBenchmark", "Measure the throughput of neural inference");
    options.add_options()
    ("c,content", "Content", cxxopts::value<std::string>()->default_value("lol0"))
    // ("r,resolution", "Resolution", cxxopts::value<int>()->default_value("2160"))
    ("i,instance", "Instance", cxxopts::value<std::string>())
    ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string content = result["content"].as<std::string>();
    int resolution = 2160;
    std::string instance_name = result["instance"].as<std::string>();

    // profile
    int qp = 95;
    int num_workers = 2;
    JPEGProfile profile = JPEGProfile(qp, num_workers);

    // load rgb images
    int height = 2160;
    int width = 3840;
    int stream_id = 0;
    std::vector<std::vector<EngorgioFrame*>> frames_per_request;
    EngorgioFrame* frame;
    std::string image_dir = "/workspace/research/engorgio/dataset/" + content + "/image";
    // std::cout << image_dir << std::endl;
    std::string image_path;
    char buffer[256];

    // run test (throughput)
    // std::cout << "hello0" << std::endl;
    int num_repeats = 10;
    // int num_repeats = 3;
    for (int i = 0; i < num_repeats + NUM_DUMMY_ITERS; i++)
    {
        std::vector<EngorgioFrame*> frames;
        for (int j = 1; j <= 50; j++)
        {
            sprintf(buffer, "%d",  j);
            image_path = image_dir + "/" + std::string(buffer) + ".rgb";
            // std::cout << image_path << std::endl;
            frame = LoadRGBFrame(j, width, height, image_path);
            frames.push_back(frame);
        }   
        frames_per_request.push_back(frames);
    }
    // std::cout << "hello1" << std::endl;
    std::vector<double> results = test_throughput_kakadu(profile, stream_id, frames_per_request, num_repeats);

    // Save a json file
    std::string resolution_name = std::to_string(resolution) + "p";
    fs::path json_dir = fs::path(ENGORGIO_RESULT_DIR) / "evaluation" / instance_name / resolution_name;
    if (!fs::exists(json_dir))
        fs::create_directories(json_dir);
    fs::path json_path = json_dir / "engorgio_encode_result.json";

    json object;
    object["latency"] = results[0];
    object["num_frames"] = results[1];
    object["throughput"] = results[2];

    std::ofstream json_file(json_path);
    json_file << std::setw(4) << object << std::endl;

    return 0;
}