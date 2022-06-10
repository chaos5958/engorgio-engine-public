#pragma once

enum class NEError : int
{
	kSuccess = 0,
	kGPUError = 1,
	kFileError = 2,
	kRuntimeError = 3
};

const size_t GB_IN_BYTES = (1 << 30);
const size_t MB_IN_BYTES = (1 << 20);

// TODO: host buffers