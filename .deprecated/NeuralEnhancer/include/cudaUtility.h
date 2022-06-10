#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

/** Convert \p size uint8_t device array \p src to float device array \p dst asynchronously */
void uchar2floatArray(uint8_t* src, uint32_t* dst, size_t size, cudaStream_t stream);

/** Convert \p size uint8_t device array \p src to half float device array \p dst asynchronously */
void uchar2halfArray(uint8_t* src, uint16_t* dst, size_t size, cudaStream_t stream);

/** Convert \p size float device array \p src to uint8_t device array \p dst asynchronously */
void float2ucharArray(uint32_t* src, uint8_t* dst, size_t size, cudaStream_t stream);

/** Convert \p size half float device array \p src to uint8_t device array \p dst asynchronously */
void half2ucharArray(uint16_t* src, uint8_t* dst, size_t size, cudaStream_t stream);