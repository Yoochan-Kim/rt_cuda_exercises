// Stage 8: Early-Exit Warp Divergence
// Benchmarks divergent and stream-compaction variants for early-exit workloads.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cuda_utils.cuh"

inline constexpr int kThreadsPerBlock = 512;
inline constexpr float kEnergyThreshold = 0.7f;
inline constexpr int kComputeIterations = 1000;
using StageValue = float;

struct Particle {
  float energy;
  float x, y, z;
  float vx, vy, vz;
};

// Expensive computation simulating physics update
__device__ float expensiveComputation(float energy, float x, float y, float z) {
  float result = 0.0f;
  for (int i = 0; i < kComputeIterations; ++i) {
    float temp = energy + x * sinf(y + i * 0.01f);
    result += temp * cosf(z + i * 0.01f);
    result = result * 0.99f + temp * 0.01f;
  }
  return result;
}

/* Divergent baseline with early-exit.
 * Particles below threshold exit early, causing warp divergence.
 */
__global__ void divergentKernel(const Particle* input,
                                StageValue* output,
                                std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  Particle p = input[idx];

  if (p.energy < kEnergyThreshold) {
    output[idx] = 0.0f;
    return;
  }

  float result = expensiveComputation(p.energy, p.x, p.y, p.z);
  output[idx] = result;
}

/* TODO:
 * Stream compaction phase 1 - Filter active particles:
 *   - For particles with energy >= kEnergyThreshold:
 *     - Use atomicAdd on outputCount to get unique position
 *     - Write particle to output[pos]
 *     - Write original index to indices[pos]
 *   - Guard against idx >= count.
 */
__global__ void compactKernel(const Particle* input,
                              Particle* output,
                              int* outputCount,
                              int* indices,
                              std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  Particle p = input[idx];

  if (p.energy >= kEnergyThreshold) {
    int pos = atomicAdd(outputCount, 1);
    output[pos] = p;
    indices[pos] = idx;
  }
}

/* TODO:
 * Stream compaction phase 2 - Process compacted data:
 *   - Read particle from input[idx]
 *   - Run expensiveComputation()
 *   - Write result to output[indices[idx]] (original position)
 *   - Guard against idx >= count.
 */
__global__ void processCompactedKernel(const Particle* input,
                                       StageValue* output,
                                       const int* indices,
                                       std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  Particle p = input[idx];
  float result = expensiveComputation(p.energy, p.x, p.y, p.z);

  int originalIdx = indices[idx];
  output[originalIdx] = result;
}

cudaError_t launchDivergentKernel(const Particle* deviceInput,
                                   StageValue* deviceOutput,
                                   std::size_t count) {
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(gridSizeForCount(count, kThreadsPerBlock));
  divergentKernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, count);
  return cudaGetLastError();
}

cudaError_t launchCompactionKernels(const Particle* deviceInput,
                                     Particle* deviceCompacted,
                                     int* deviceCount,
                                     int* deviceIndices,
                                     StageValue* deviceOutput,
                                     std::size_t count) {
  // Reset count
  cudaMemset(deviceCount, 0, sizeof(int));

  // Phase 1: Filter and compact active particles
  const dim3 blockDim(kThreadsPerBlock);
  const dim3 gridDim(gridSizeForCount(count, kThreadsPerBlock));
  compactKernel<<<gridDim, blockDim>>>(deviceInput, deviceCompacted,
                                       deviceCount, deviceIndices, count);

  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  // Get compacted count (how many active particles)
  int hostCount = 0;
  status = cudaMemcpy(&hostCount, deviceCount, sizeof(int),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return status;

  if (hostCount == 0) return cudaSuccess;

  // Phase 2: Process densely-packed active particles
  const dim3 gridDim2(gridSizeForCount(hostCount, kThreadsPerBlock));
  processCompactedKernel<<<gridDim2, blockDim>>>(deviceCompacted, deviceOutput,
                                                  deviceIndices, hostCount);

  return cudaGetLastError();
}