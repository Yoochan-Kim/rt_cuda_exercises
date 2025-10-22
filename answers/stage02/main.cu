// Stage 2 exercise: verify GPU vector addition against a CPU reference.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"

constexpr int kBlocks = 8;
constexpr int kThreads = 256;
constexpr int kElementCount = kBlocks * kThreads;

#include "todo.cu"

#ifndef SKIP_CPU
bool compareVectors(const std::vector<float>& gpu,
                    const std::vector<float>& reference,
                    int& mismatchIndex,
                    float& gpuValue,
                    float& refValue) {
    constexpr float kTolerance = 1e-5f;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        if (std::fabs(gpu[i] - reference[i]) > kTolerance) {
            mismatchIndex = static_cast<int>(i);
            gpuValue = gpu[i];
            refValue = reference[i];
            return false;
        }
    }
    return true;
}
#endif

int main() {
    std::vector<float> hostA(kElementCount);
    std::vector<float> hostB(kElementCount);
    std::vector<float> hostC(kElementCount, 0.0f);
#ifndef SKIP_CPU
    std::vector<float> reference(kElementCount, 0.0f);
#endif

    for (int i = 0; i < kElementCount; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(kElementCount - i);
#ifndef SKIP_CPU
        reference[i] = hostA[i] + hostB[i];
#endif
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    CHECK_CUDA(cudaMalloc(&deviceA, kElementCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceB, kElementCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceC, kElementCount * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(deviceA, hostA.data(),
                          kElementCount * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceB, hostB.data(),
                          kElementCount * sizeof(float), cudaMemcpyHostToDevice));

    vectorAddKernel<<<kBlocks, kThreads>>>(deviceA, deviceB, deviceC, kElementCount);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hostC.data(), deviceC,
                          kElementCount * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(deviceC));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceA));

#ifdef SKIP_CPU
    // Print all GPU results directly
    for (int i = 0; i < kElementCount; ++i) {
        std::cout << hostC[i] << std::endl;
    }
#else
    int mismatchIndex = -1;
    float gpuValue = 0.0f;
    float refValue = 0.0f;
    if (compareVectors(hostC, reference, mismatchIndex, gpuValue, refValue)) {
        std::cout << "Stage 2 vector add matches reference ✅" << std::endl;
    } else {
        const float diff = std::fabs(gpuValue - refValue);
        std::cout << "Stage 2 vector add mismatch ❌" << std::endl;
        std::cout << "  first difference at index " << mismatchIndex << '\n'
                  << "  GPU value: " << gpuValue << '\n'
                  << "  REF value: " << refValue << '\n'
                  << "  |diff|:    " << diff << std::endl;
    }
#endif

    return 0;
}
