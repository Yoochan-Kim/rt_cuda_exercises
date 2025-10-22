#pragma once

// Simple helper to make CUDA runtime error checking readable.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

