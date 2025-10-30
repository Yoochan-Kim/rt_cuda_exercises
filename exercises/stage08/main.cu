// Stage 8: Early-Exit Warp Divergence
// Benchmarks divergent and stream-compaction variants for early-exit workloads.

#include <cstddef>
#include <vector>
#include <random>
#include <iomanip>
#include <iostream>

#include "cuda_utils.cuh"

#include "todo.cu"

constexpr std::size_t kElementCount = 1 << 24;  // 1M particles
constexpr int kBenchmarkRepeats = 100;

void generateParticles(std::vector<Particle>& particles) {
  std::mt19937 rng(12345u);
  std::uniform_real_distribution<float> energyDist(0.0f, 1.0f);
  std::uniform_real_distribution<float> posDist(-10.0f, 10.0f);
  std::uniform_real_distribution<float> velDist(-1.0f, 1.0f);

  for (Particle& p : particles) {
    p.energy = energyDist(rng);
  }
  for (Particle& p : particles) {
    p.x = posDist(rng);
  }
  for (Particle& p : particles) {
    p.y = posDist(rng);
  }
  for (Particle& p : particles) {
    p.z = posDist(rng);
  }
  for (Particle& p : particles) {
    p.vx = velDist(rng);
    p.vy = velDist(rng);
    p.vz = velDist(rng);
  }
}

int main() {
  std::vector<Particle> particles(kElementCount);
  generateParticles(particles);

  const std::size_t count = particles.size();
  const std::size_t particleBytes = count * sizeof(Particle);
  const std::size_t outputBytes = count * sizeof(StageValue);

  // Allocate device memory
  Particle* deviceInput = nullptr;
  StageValue* deviceOutput = nullptr;
  Particle* deviceCompacted = nullptr;
  int* deviceCount = nullptr;
  int* deviceIndices = nullptr;

  cudaError_t status = cudaMalloc(&deviceInput, particleBytes);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  status = cudaMalloc(&deviceOutput, outputBytes);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaMalloc(&deviceCompacted, particleBytes);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaMalloc(&deviceCount, sizeof(int));
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceCompacted);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaMalloc(&deviceIndices, count * sizeof(int));
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceCount);
    cudaFree(deviceCompacted);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  // Create events for timing
  cudaEvent_t startEvent = nullptr;
  cudaEvent_t stopEvent = nullptr;

  status = cudaEventCreate(&startEvent);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceIndices);
    cudaFree(deviceCount);
    cudaFree(deviceCompacted);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaEventCreate(&stopEvent);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaEventDestroy(startEvent);
    cudaFree(deviceIndices);
    cudaFree(deviceCount);
    cudaFree(deviceCompacted);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  // Copy input data
  status = cudaMemcpy(deviceInput, particles.data(), particleBytes,
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::cerr << "Stage 8 failed: " << cudaGetErrorString(status) << std::endl;
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);
    cudaFree(deviceIndices);
    cudaFree(deviceCount);
    cudaFree(deviceCompacted);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  std::vector<StageValue> hostOutput(count);
  double divergentTime = 0.0;
  double compactionTime = 0.0;
  double divergentChecksum = 0.0;
  double compactionChecksum = 0.0;

  const char* failedLabel = nullptr;

  // Test 1: Divergent kernel with early-exit
  cudaMemset(deviceOutput, 0, outputBytes);
  status = launchDivergentKernel(deviceInput, deviceOutput, count);
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(startEvent);
  }
  if (status == cudaSuccess) {
    for (int iter = 0; iter < kBenchmarkRepeats; ++iter) {
      launchDivergentKernel(deviceInput, deviceOutput, count);
    }
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(stopEvent);
  }
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(stopEvent);
  }
  if (status == cudaSuccess) {
    float elapsedMs = 0.0f;
    status = cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    if (status == cudaSuccess) {
      divergentTime = static_cast<double>(elapsedMs) / kBenchmarkRepeats;
    }
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(hostOutput.data(), deviceOutput, outputBytes,
                        cudaMemcpyDeviceToHost);
  }
  if (status == cudaSuccess) {
    double sum = 0.0;
    for (StageValue v : hostOutput) {
      sum += static_cast<double>(v);
    }
    divergentChecksum = sum;
  }
  if (status != cudaSuccess) {
    failedLabel = "divergent";
  }

  // Test 2: Stream compaction
  if (failedLabel == nullptr) {
    cudaMemset(deviceOutput, 0, outputBytes);
    status = launchCompactionKernels(deviceInput, deviceCompacted, deviceCount,
                                      deviceIndices, deviceOutput, count);
    if (status == cudaSuccess) {
      status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
      status = cudaEventRecord(startEvent);
    }
    if (status == cudaSuccess) {
      for (int iter = 0; iter < kBenchmarkRepeats; ++iter) {
        launchCompactionKernels(deviceInput, deviceCompacted, deviceCount,
                                deviceIndices, deviceOutput, count);
      }
      status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
      status = cudaEventRecord(stopEvent);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(stopEvent);
    }
    if (status == cudaSuccess) {
      float elapsedMs = 0.0f;
      status = cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
      if (status == cudaSuccess) {
        compactionTime = static_cast<double>(elapsedMs) / kBenchmarkRepeats;
      }
    }
    if (status == cudaSuccess) {
      status = cudaMemcpy(hostOutput.data(), deviceOutput, outputBytes,
                          cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
      double sum = 0.0;
      for (StageValue v : hostOutput) {
        sum += static_cast<double>(v);
      }
      compactionChecksum = sum;
    }
    if (status != cudaSuccess) {
      failedLabel = "compaction";
    }
  }

  // Cleanup
  cudaEventDestroy(stopEvent);
  cudaEventDestroy(startEvent);
  cudaFree(deviceIndices);
  cudaFree(deviceCount);
  cudaFree(deviceCompacted);
  cudaFree(deviceOutput);
  cudaFree(deviceInput);

  if (failedLabel != nullptr) {
    std::cerr << "Stage 8 " << failedLabel
              << " variant failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

#ifdef SKIP_CPU
  std::cout << std::setprecision(10) << divergentChecksum << '\n'
            << compactionChecksum << std::endl;
  std::cerr << "Divergent time avg (ms): " << divergentTime << std::endl;
  std::cerr << "Compaction time avg (ms): " << compactionTime << std::endl;
#else
  std::cout << "Stage 8 early-exit divergence ✅" << std::endl;
  std::cout << "\nDataset:" << std::endl;
  std::cout << "  Particles: " << count << std::endl;
  std::cout << "  Block size: " << kThreadsPerBlock << std::endl;
  std::cout << "  Energy threshold: " << kEnergyThreshold << std::endl;
  std::cout << "  Compute iterations: " << kComputeIterations << std::endl;

  // Calculate active percentage
  int activeCount = 0;
  for (const Particle& p : particles) {
    if (p.energy >= kEnergyThreshold) {
      activeCount++;
    }
  }
  float activePercent = 100.0f * activeCount / count;

  std::cout << "  Active particles: " << activeCount << " (" << std::fixed
            << std::setprecision(1) << activePercent << "%)" << std::endl;

  std::cout << "\nResults (avg over " << kBenchmarkRepeats << " runs):"
            << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Divergent (early-exit) : " << divergentTime << " ms | checksum "
            << std::setprecision(2) << divergentChecksum << std::setprecision(3)
            << std::endl;
  std::cout << "  Stream compaction      : " << compactionTime << " ms | checksum "
            << std::setprecision(2) << compactionChecksum << std::setprecision(3)
            << std::endl;

  std::cout << "\nSpeedup:" << std::endl;
  std::cout << std::setprecision(2);
  std::cout << "  Stream compaction : " << (divergentTime / compactionTime) << "x faster"
            << std::endl;
#endif

  return 0;
}
