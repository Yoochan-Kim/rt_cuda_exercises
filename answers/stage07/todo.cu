// Stage 7: Warp Divergence Benchmark
// Benchmarks divergent and split-kernel variants for parity-dependent workloads.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "cuda_utils.cuh"

inline constexpr int kThreadsPerBlock = 512;
inline constexpr int kHeavyIterations = 96;
inline constexpr int kLightIterations = 8;

__device__ float heavyIteration(float value) {
  return sinf(value) * cosf(value) + 0.5f;
}

__device__ float lightIteration(float value) {
  constexpr float kScale = 1.125f;
  constexpr float kBias = 0.25f;
  return value * kScale + kBias;
}

/* TODO:
 * Divergent baseline:
 *   - Even indices execute kHeavyIterations of heavyIteration().
 *   - Odd indices execute kLightIterations of lightIteration().
 *   - Guard against idx >= count.
 */
__global__ void divergentKernel(const float* input,
                                float* output,
                                std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  float value = input[idx];
  if ((idx & 1u) == 0u) {
    for (int iter = 0; iter < kHeavyIterations; ++iter) {
      value = heavyIteration(value);
    }
  } else {
    for (int iter = 0; iter < kLightIterations; ++iter) {
      value = lightIteration(value);
    }
  }
  output[idx] = value;
}

/* TODO:
 * Split kernels (leave larger gaps for the final optimisation exercise).
 */
__global__ void evenKernel(const float* input,
                           float* output,
                           std::size_t count) {
  const std::size_t logicalIdx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t idx = logicalIdx * 2u;
  if (idx >= count) {
    return;
  }

  float value = input[idx];
  for (int iter = 0; iter < kHeavyIterations; ++iter) {
    value = heavyIteration(value);
  }
  output[idx] = value;
}

__global__ void oddKernel(const float* input,
                          float* output,
                          std::size_t count) {
  const std::size_t logicalIdx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t idx = logicalIdx * 2u + 1u;
  if (idx >= count) {
    return;
  }

  float value = input[idx];
  for (int iter = 0; iter < kLightIterations; ++iter) {
    value = lightIteration(value);
  }
  output[idx] = value;
}

cudaError_t launchDivergentKernel(const float* deviceInput,
                                  float* deviceOutput,
                                  std::size_t count) {
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(gridSizeForCount(count, kThreadsPerBlock));
  divergentKernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, count);
  return cudaGetLastError();
}

cudaError_t launchSplitKernels(const float* deviceInput,
                               float* deviceOutput,
                               std::size_t count) {
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(gridSizeForCount((count + 1) / 2, kThreadsPerBlock));
  evenKernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, count);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  oddKernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, count);
  return cudaGetLastError();
}
