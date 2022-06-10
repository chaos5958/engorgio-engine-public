#include "cudaUtility.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 512

__global__ void uchar2float(uint8_t* src, float* dst, int maxIndex) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = min(index, maxIndex);
    dst[index] = __int2float_rd(src[index]);
}

__global__ void uchar2half(uint8_t* src, half* dst, int maxIndex) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = min(index, maxIndex);
    dst[index] = __int2half_rd(src[index]);
}

__global__ void float2uchar(float* src, uint8_t* dst, int maxIndex) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = min(index, maxIndex);
    dst[index] = max(min(__float2int_rd(src[index]), 255), 0);
}

__global__ void half2uchar(half* src, uint8_t* dst, int maxIndex) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    index = min(index, maxIndex);
    dst[index] = max(min(__half2int_rd(src[index]), 255), 0);
}

void uchar2floatArray(uint8_t* src, uint32_t* dst, size_t size, cudaStream_t stream) {
    uchar2float << < (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream >> > (src, (float*)dst, (int)size);
}

void uchar2halfArray(uint8_t* src, uint16_t* dst, size_t size, cudaStream_t stream) {
    uchar2half << < (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream >> > (src, (half*)dst, (int)size);
}

void float2ucharArray(uint32_t* src, uint8_t* dst, size_t size, cudaStream_t stream) {
    float2uchar << < (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream >> > ((float*)src, dst, (int)size);
}

void half2ucharArray(uint16_t* src, uint8_t* dst, size_t size, cudaStream_t stream) {
    half2uchar << < (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream >> > ((half*)src, dst, (int)size);
}