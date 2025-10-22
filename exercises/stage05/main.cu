// Stage 5: Matrix Multiplication without Shared Memory
// This demonstrates a naive matrix multiplication implementation.

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix.h"

#include "todo.cu"

// Matrix dimensions (multiples of BLOCK_SIZE=16)
constexpr int M = 5120;  // A: M x K, C: M x N
constexpr int K = 2048;  // A: M x K, B: K x N
constexpr int N = 3072;  // B: K x N, C: M x N
constexpr float kTolerance = 1e-3f;

void fillMatrix(Matrix matrix, float scale) {
    for (int row = 0; row < matrix.height; ++row) {
        for (int col = 0; col < matrix.width; ++col) {
            const int idx = row * matrix.width + col;
            matrix.elements[idx] = scale * static_cast<float>(idx % 100);
        }
    }
}

#ifndef SKIP_CPU
void multiplyMatricesHost(const Matrix& a, const Matrix& b, Matrix& out) {
    // C[i][j] = sum of A[i][k] * B[k][j] for k = 0 to a.width-1
    // A is M x K, B is K x N, C is M x N
    for (int row = 0; row < a.height; ++row) {
        for (int col = 0; col < b.width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < a.width; ++k) {
                sum += a.elements[row * a.width + k] * b.elements[k * b.width + col];
            }
            out.elements[row * b.width + col] = sum;
        }
    }
}

bool compareMatrices(const Matrix& lhs, const Matrix& rhs, float tolerance) {
    const std::size_t count = matrixElementCount(lhs);
    int errors = 0;
    constexpr int maxPrintErrors = 10;

    for (std::size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(lhs.elements[i] - rhs.elements[i]);
        const float relativeError = diff / std::fmax(std::fabs(rhs.elements[i]), 1e-6f);

        if (diff > tolerance && relativeError > tolerance) {
            if (errors < maxPrintErrors) {
                std::cerr << "Mismatch at element " << i
                          << " lhs=" << lhs.elements[i]
                          << " rhs=" << rhs.elements[i]
                          << " diff=" << diff
                          << " rel_err=" << relativeError << '\n';
            }
            ++errors;
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << count
                  << " (" << (100.0 * errors / count) << "%)\n";
        return false;
    }
    return true;
}
#endif

int runMatrixMulDemo() {
    std::vector<float> hostAData(M * K, 0.0f);
    std::vector<float> hostBData(K * N, 0.0f);
    std::vector<float> hostCData(M * N, 0.0f);
#ifndef SKIP_CPU
    std::vector<float> hostReferenceData(M * N, 0.0f);
#endif

    Matrix hostA{K, M, hostAData.data()};  // A: M x K (width=K, height=M)
    Matrix hostB{N, K, hostBData.data()};  // B: K x N (width=N, height=K)
    Matrix hostC{N, M, hostCData.data()};  // C: M x N (width=N, height=M)
#ifndef SKIP_CPU
    Matrix reference{N, M, hostReferenceData.data()};
#endif

    fillMatrix(hostA, 0.01f);
    fillMatrix(hostB, 0.02f);

#ifndef SKIP_CPU
    // CPU timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    multiplyMatricesHost(hostA, hostB, reference);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
#endif

    // GPU timing with cudaEvent
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(multiplyMatricesOnDevice(hostA, hostB, hostC));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

#ifdef SKIP_CPU
    // Print all GPU results directly
    const std::size_t count = matrixElementCount(hostC);
    for (std::size_t i = 0; i < count; ++i) {
        std::cout << hostC.elements[i] << std::endl;
    }
#else
    // Verification
    if (compareMatrices(hostC, reference, kTolerance)) {
        std::cout << "Stage 5 matrix multiplication matches reference ✅" << std::endl;
    } else {
        std::cout << "Stage 5 matrix multiplication mismatch ❌" << std::endl;
    }

    // Performance comparison
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << "  CPU Time: " << cpuTime << " ms" << std::endl;
    std::cout << "  GPU Time: " << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpuTime / gpuTime) << "x" << std::endl;
#endif

    return 0;
}

int main() {
    return runMatrixMulDemo();
}
