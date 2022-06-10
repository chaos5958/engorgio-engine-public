#pragma once

#include <string>
#include "BaseModel.h"

struct VideoType
{
	int width;
	int height;

	bool operator==(const VideoType &o) const {
        return width == o.width && height == o.height;
    }

    bool operator<(const VideoType &o)  const {
        return height < o.height || (height == o.height && width < o.width);
    }
};

struct DNNType
{
	int num_blocks;
	int num_features;
	int scale;
};

inline std::string get_dnn_name(const std::string& mname, const BuildType& btype, const DNNType dtype)
{
	std::string model, format;

	model = mname + "_B" + std::to_string(dtype.num_blocks) + "_F" + std::to_string(dtype.num_features) + "_S" + std::to_string(dtype.scale);
	switch (btype)
	{
	case BuildType::kTRT:
		format = ".plan";
		break;
	case BuildType::kONNX:
		format = ".onnx";
		break;
	case BuildType::kPyTorch:
		format = ".pt";
		break;
	default:
		throw std::runtime_error("Invalid BuildType");
	}
	
	return std::string(model + format);
}

std::string get_dnn_file(const std::string& dir, const VideoType& vtype, const std::string& dnn_name)
{
	return std::string(dir + std::to_string(vtype.height) + "p" + "/" + dnn_name);
}

std::string get_btype_name(BuildType btype)
{
	std::string name;

	switch (btype)
	{
	case BuildType::kONNX:
		name = "onnx";
		break;
	case BuildType::kTRT:
		name = "trt";
		break;
	case BuildType::kPyTorch:
		name = "pytorch";
		break;
	}
	return name;
}

std::string get_mtype_name(MemoryType mtype)
{
	std::string name;

	switch (mtype)
	{
	case MemoryType::kAllocate:
		name = "allocate";
		break;
	case MemoryType::kPreallocate:
		name = "preallocate";
		break;
	}
	return name;
}