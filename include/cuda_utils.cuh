#pragma once

// Common CUDA utilities for error checking and kernel launch helpers.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Error checking helper
inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

// Grid size calculation helper
// Calculates number of blocks needed for given element count and block size.
// Returns at least 1 block even for empty input.
template<typename T>
inline unsigned int gridSizeForCount(std::size_t count, T threadsPerBlock) {
  if (count == 0) {
    return 1u;
  }
  return static_cast<unsigned int>((count + threadsPerBlock - 1) / threadsPerBlock);
}

