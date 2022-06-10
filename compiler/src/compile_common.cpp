#include <filesystem>
#include <iostream>
#include <fstream>
#include "compile_common.h"

namespace fs = std::filesystem;

OnnxModel::OnnxModel(const std::string& path, const std::string& name)
{
    if (!fs::exists(path))
        throw std::runtime_error("Model does not exist");
    path_ = path;
    name_ = name;
    is_loaded_ = false;
}
// destructor
OnnxModel::~OnnxModel()
{}

bool OnnxModel::Load()
{
    std::ifstream onnx_file(path_, std::ios::binary | std::ios::ate);
    if (!onnx_file.is_open())
    {
        return false;
    }
    std::streamsize file_size = onnx_file.tellg();

    buf_ = std::vector<char>(file_size);

    onnx_file.seekg(0, std::ios::beg);
    if (!onnx_file.read(buf_.data(), buf_.size()))
    {
        return false;
    }

    // std::cout << buf_.size() << std::endl;
    is_loaded_ = true;
    return true;
}

char* OnnxModel::GetModel()
{
    if (!is_loaded_)
        return nullptr;
    return buf_.data();
}

size_t OnnxModel::GetSize()
{
    if (!is_loaded_)
        return 0;
    return buf_.size();
}


// void OnnxModel::RegisterName(const std::string& name)
// {
//     model_name_ = name;
// }

// const std::string OnnxModel::ExtractNameFromFile()
// {
//     std::stringstream ss(file_);
    
//     std::getline(ss, model_name_, '.');
//     if (model_name_.empty())
//     {
//         std::cout << "Failed to parse the file name " << file_ << std::endl;
//         return std::string{};
//     }
//     return model_name_;
// }

// const std::string OnnxModel::GetModelName()
// {
//     return model_name_;
// }
