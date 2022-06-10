#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "compile_engine.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cxxopts.hpp"
#include "tool_common.h"

namespace fs = std::filesystem;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

bool onnx_to_plan(std::string &onnx_path, std::string &plan_path)
{
    std::cout << "onnx path: " << onnx_path << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    OnnxModel *onnx_model = new OnnxModel(onnx_path, std::string("tmp"));
    if (!onnx_model->Load())
        return false;


    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // auto profile = builder->createOptimizationProfile();
    // profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 360, 640));
    // profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 360, 640));
    // profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 360, 640));
    // profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 720, 1280))/;
    // profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 1080, 1920));

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
        
    // config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(4 * size_t(1<<30)); // TODO: configure this
    std::cout << "Workspace size: " << config->getMaxWorkspaceSize() << std::endl;
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kREFIT); // TODO 
    // config->addOptimizationProfile(profile);

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

    auto parsed = parser->parse(onnx_model->GetModel(), onnx_model->GetSize());  
    if (!parsed)
    {
        return false;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!engine)
    {
        return false;
    }

    auto serialize = engine->serialize();

    std::ofstream fout;
    fout.open(plan_path, std::ios::out | std::ios::binary);
    if (!fout.is_open())
    {   
        return false;
    }
    fout.write((const char*)serialize->data(), serialize->size());
    fout.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "onnx_to_plan: " << elapsed.count() << "s" << std::endl;

    return true;
}

// TODO: command line parsing (using cxxopts)
int main(int argc, char** argv)
{
    cxxopts::Options options("ONNXConverter", "Convert a ONNX model to a TRT serialized engine");

    options.add_options()
    ("c,content", "Content", cxxopts::value<std::string>()->default_value("all"))
    ("d,duration", "Duration", cxxopts::value<int>()->default_value("600"))
    ("r,resolution", "Input resolution", cxxopts::value<int>()->default_value("720"))
    ("m,model", "Model", cxxopts::value<std::string>()->default_value("EDSR_B8_F32_S3"))
    ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string data_dir = ENGORGIO_RESULT_DIR;
    std::string content = result["content"].as<std::string>();
    int resolution = result["resolution"].as<int>();
    int duration = result["duration"].as<int>();
    std::string model = result["model"].as<std::string>();
    std::string video = get_video_name(resolution, duration);
    
    std::string onnx_path, plan_path;
    if (content == std::string("all"))
    {
        for (auto &content : ENGORGIO_CONTENTS)
        {
            onnx_path = get_onnx_path(content, model, resolution, duration);
            plan_path = get_trt_path(content, model, resolution, duration);
            assert(std::filesystem::exists(onnx_path));
        }
    }
    else
    {
        onnx_path = get_onnx_path(content, model, resolution, duration);
        plan_path = get_trt_path(content, model, resolution, duration);
        assert(std::filesystem::exists(onnx_path));
    }

    if (content == std::string("all"))
    {
        for (auto &content : ENGORGIO_CONTENTS)
        {
            onnx_path = get_onnx_path(content, model, resolution, duration);
            plan_path = get_trt_path(content, model, resolution, duration);
            if (!fs::exists(fs::path(plan_path).parent_path()))
                fs::create_directories(fs::path(plan_path).parent_path());  
            onnx_to_plan(onnx_path, plan_path);
        }
    }
    else
    {
        onnx_path = get_onnx_path(content, model, resolution, duration);
        plan_path = get_trt_path(content, model, resolution, duration);
        if (!fs::exists(fs::path(plan_path).parent_path()))
            fs::create_directories(fs::path(plan_path).parent_path());   
        onnx_to_plan(onnx_path, plan_path);
    }

    return 0;
}