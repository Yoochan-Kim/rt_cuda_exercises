// Stage 7: Warp Divergence Benchmark
// Provides divergent and split-kernel variants for parity-dependent workloads.

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
  // TODO: implement divergent kernel described above.
  (void)input;
  (void)output;
  (void)count;
}

__global__ void evenKernel(const float* input,
                           float* output,
                           std::size_t count) {
  const std::size_t threadIndex =
      blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t idx = threadIndex * 2;
  if (idx >= count) {
    return;
  }

  float value = input[idx];
  for (int i = 0; i < kHeavyIterations; ++i) {
    value = heavyIteration(value);
  }
  output[idx] = value;
}

__global__ void oddKernel(const float* input,
                          float* output,
                          std::size_t count) {
  const std::size_t threadIndex =
      blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t idx = threadIndex * 2 + 1;
  if (idx >= count) {
    return;
  }

  float value = input[idx];
  for (int i = 0; i < kLightIterations; ++i) {
    value = lightIteration(value);
  }
  output[idx] = value;
}

inline unsigned int blocksForCount(std::size_t count) {
  if (count == 0) {
    return 1u;
  }
  return static_cast<unsigned int>((count + kThreadsPerBlock - 1) /
                                   kThreadsPerBlock);
}

cudaError_t launchDivergentKernel(const float* deviceInput,
                                  float* deviceOutput,
                                  std::size_t count) {
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(blocksForCount(count));
  // TODO: launch divergentKernel with gridDim/blockDim.
  (void)deviceInput;
  (void)deviceOutput;
  (void)count;
  return cudaErrorNotSupported;
}

cudaError_t launchSplitKernels(const float* deviceInput,
                               float* deviceOutput,
                               std::size_t count) {
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(blocksForCount((count + 1) / 2));
  // TODO: launch evenKernel and oddKernel back-to-back using the grid config above.
  (void)deviceInput;
  (void)deviceOutput;
  (void)count;
  return cudaErrorNotSupported;
}
