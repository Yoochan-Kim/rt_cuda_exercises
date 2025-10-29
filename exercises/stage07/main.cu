// Stage 7: Shared Memory Reduction Baseline
// Baseline implementation that loads into shared memory before performing interleaved-addressing reduction.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "cuda_utils.cuh"

#include "todo.cu"

constexpr std::size_t kElementCount = 1 << 24;
constexpr float kRelativeTolerance = 1e-5f;
constexpr int kGpuRuns = 100;

#ifndef SKIP_CPU
double reduceOnHost(const std::vector<float>& values) {
    double sum = 0.0;
    for (float v : values) {
        sum += static_cast<double>(v);
    }
    return sum;
}
#endif

void fillInput(std::vector<float>& values) {
    std::mt19937 rng(12345u);
    constexpr float kInvMax = 1.0f / static_cast<float>(std::numeric_limits<std::uint32_t>::max());
    for (float& v : values) {
        const std::uint32_t raw = rng();
        const float normalized = static_cast<float>(raw) * kInvMax;
        v = normalized * 2.0f - 1.0f;
    }
}

int runReductionDemo() {
    std::vector<float> hostInput(kElementCount);
    fillInput(hostInput);

    float gpuSum = 0.0f;

#ifndef SKIP_CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    double cpuSum = reduceOnHost(hostInput);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
#endif

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double gpuTimeTotal = 0.0;
    double gpuTimeMin = std::numeric_limits<double>::infinity();
    double gpuTimeMax = 0.0;

    for (int run = 0; run < kGpuRuns; ++run) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(reduceSharedMemoryBaseline(hostInput.data(), hostInput.size(), &gpuSum));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float runTimeMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&runTimeMs, start, stop));

        gpuTimeTotal += static_cast<double>(runTimeMs);
        if (runTimeMs < gpuTimeMin) {
            gpuTimeMin = static_cast<double>(runTimeMs);
        }
        if (runTimeMs > gpuTimeMax) {
            gpuTimeMax = static_cast<double>(runTimeMs);
        }
    }

    const double gpuTimeAvg = gpuTimeTotal / static_cast<double>(kGpuRuns);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

#ifdef SKIP_CPU
    std::cout << gpuSum << std::endl;
    std::cerr << "GPU time (avg over " << kGpuRuns << " runs) : " << gpuTimeAvg << " ms" << std::endl;
    std::cerr << "GPU time (min / max) : " << gpuTimeMin << " ms / " << gpuTimeMax << " ms" << std::endl;
#else
    const double gpuSumAsDouble = static_cast<double>(gpuSum);
    const double denom = (std::abs(cpuSum) > 1e-12) ? std::abs(cpuSum) : 1e-12;
    const double relativeError = std::abs(cpuSum - gpuSumAsDouble) / denom;

    if (relativeError <= kRelativeTolerance) {
        std::cout << "Stage 7 reduction matches reference ✅" << std::endl;
    } else {
        std::cout << "Stage 7 reduction mismatch ❌" << std::endl;
    }

    std::cout << "\nInput size: " << hostInput.size() << " elements" << std::endl;
    std::cout << "CPU sum : " << cpuSum << std::endl;
    std::cout << "GPU sum : " << gpuSumAsDouble << std::endl;
    std::cout << "Relative error: " << relativeError << std::endl;

    std::cout << "\nTiming:" << std::endl;
    std::cout << "  CPU time : " << cpuTime << " ms" << std::endl;
    std::cout << "  GPU time (avg over " << kGpuRuns << " runs) : " << gpuTimeAvg << " ms" << std::endl;
    std::cout << "  GPU time (min / max) : " << gpuTimeMin << " ms / " << gpuTimeMax << " ms" << std::endl;
#endif

    return 0;
}

int main() {
    return runReductionDemo();
}
