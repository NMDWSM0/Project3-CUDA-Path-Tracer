#pragma once

#include <device_launch_parameters.h>
#include <device_functions.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define GET_BIT(num, k) (((num) >> (k)) & 1)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

__host__ __device__ __forceinline__ constexpr int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

__host__ __device__ __forceinline__ constexpr int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}