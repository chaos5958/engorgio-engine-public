#pragma once

#include <string>
#include <vector>

struct OnnxModel
{
    std::string path_;
    std::string name_;
    bool is_loaded_;
    std::vector<char> buf_;

    OnnxModel(const std::string& path, const std::string& name);
    ~OnnxModel();
    bool Load();
    char* GetModel();
    size_t GetSize();
};

