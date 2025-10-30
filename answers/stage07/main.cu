// Stage 7: Warp Divergence Benchmark
// Benchmarks divergent and split-kernel variants for parity-dependent workloads.

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "cuda_utils.cuh"

#include "todo.cu"

constexpr std::size_t kElementCount = 1 << 24;
constexpr int kBenchmarkRepeats = 100;

void fillInput(std::vector<float>& values) {
  std::mt19937 rng(12345u);
  constexpr float kInvMax =
      1.0f / static_cast<float>(std::numeric_limits<std::uint32_t>::max());
  for (float& v : values) {
    const std::uint32_t raw = rng();
    const float normalized = static_cast<float>(raw) * kInvMax;
    v = normalized * 2.0f - 1.0f;
  }
}

int main() {
  std::vector<float> hostInput(kElementCount);
  fillInput(hostInput);

  const std::size_t count = hostInput.size();
  const std::size_t bytes = count * sizeof(float);

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;
  cudaEvent_t startEvent = nullptr;
  cudaEvent_t stopEvent = nullptr;

  cudaError_t status = cudaMalloc(&deviceInput, bytes);
  if (status != cudaSuccess) {
    std::cerr << "Stage 7 failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  status = cudaMalloc(&deviceOutput, bytes);
  if (status != cudaSuccess) {
    std::cerr << "Stage 7 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaEventCreate(&startEvent);
  if (status != cudaSuccess) {
    std::cerr << "Stage 7 failed: " << cudaGetErrorString(status) << std::endl;
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  status = cudaEventCreate(&stopEvent);
  if (status != cudaSuccess) {
    std::cerr << "Stage 7 failed: " << cudaGetErrorString(status) << std::endl;
    cudaEventDestroy(startEvent);
    cudaFree(deviceOutput);
    cudaFree(deviceInput);
    return 1;
  }

  std::vector<float> hostOutput(count, 0.0f);

  auto runVariant = [&](auto&& launchKernel,
                        double* avgTimeMs,
                        float* checksum) -> cudaError_t {
    cudaError_t localStatus = cudaMemcpy(deviceInput,
                                         hostInput.data(),
                                         bytes,
                                         cudaMemcpyHostToDevice);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }

    localStatus = launchKernel();
    if (localStatus != cudaSuccess) {
      return localStatus;
    }
    localStatus = cudaDeviceSynchronize();
    if (localStatus != cudaSuccess) {
      return localStatus;
    }

    localStatus = cudaEventRecord(startEvent);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }
    for (int iter = 0; iter < kBenchmarkRepeats; ++iter) {
      localStatus = launchKernel();
      if (localStatus != cudaSuccess) {
        return localStatus;
      }
    }
    localStatus = cudaEventRecord(stopEvent);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }
    localStatus = cudaEventSynchronize(stopEvent);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }

    float elapsedMs = 0.0f;
    localStatus = cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }

    localStatus = cudaMemcpy(hostOutput.data(),
                             deviceOutput,
                             bytes,
                             cudaMemcpyDeviceToHost);
    if (localStatus != cudaSuccess) {
      return localStatus;
    }

    double sum = 0.0;
    for (float v : hostOutput) {
      sum += static_cast<double>(v);
    }
    *avgTimeMs = static_cast<double>(elapsedMs) / kBenchmarkRepeats;
    *checksum = static_cast<float>(sum);
    return cudaSuccess;
  };

  double divergentTime = 0.0;
  double splitTime = 0.0;
  float divergentChecksum = 0.0f;
  float splitChecksum = 0.0f;

  const char* failedLabel = nullptr;

  status = runVariant(
      [&]() -> cudaError_t {
        return launchDivergentKernel(deviceInput, deviceOutput, count);
      },
      &divergentTime,
      &divergentChecksum);
  if (status != cudaSuccess) {
    failedLabel = "divergent";
  }

  if (failedLabel == nullptr) {
    status = runVariant(
        [&]() -> cudaError_t {
          return launchSplitKernels(deviceInput, deviceOutput, count);
        },
        &splitTime,
        &splitChecksum);
    if (status != cudaSuccess) {
      failedLabel = "split";
    }
  }

  cudaEventDestroy(stopEvent);
  cudaEventDestroy(startEvent);
  cudaFree(deviceOutput);
  cudaFree(deviceInput);

  if (failedLabel != nullptr) {
    std::cerr << "Stage 7 " << failedLabel
              << " variant failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

#ifdef SKIP_CPU
  std::cout << std::setprecision(10) << divergentChecksum << '\n'
            << splitChecksum << std::endl;
  std::cerr << "Divergent time avg (ms): " << divergentTime << std::endl;
  std::cerr << "Split time avg (ms): " << splitTime << std::endl;
#else
  std::cout << "Stage 7 warp divergence benchmark ✅" << std::endl;
  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Elements: " << count << std::endl;
  std::cout << "  Block size: " << kThreadsPerBlock << std::endl;
  std::cout << "  Heavy iterations: " << kHeavyIterations << std::endl;
  std::cout << "  Light iterations: " << kLightIterations << std::endl;

  std::cout << "\nResults (avg over " << kBenchmarkRepeats << " runs):"
            << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Divergent branch : " << divergentTime << " ms | checksum "
            << std::setprecision(8) << divergentChecksum << std::setprecision(3)
            << std::endl;
  std::cout << "  Split kernels    : " << splitTime << " ms | checksum "
            << std::setprecision(8) << splitChecksum << std::setprecision(3)
            << std::endl;

  std::cout << "\nSpeedup vs divergent:" << std::endl;
  const double splitSpeedup = divergentTime / splitTime;
  std::cout << std::setprecision(2);
  std::cout << "  Split: " << splitSpeedup << "x" << std::endl;
#endif

  return 0;
}
