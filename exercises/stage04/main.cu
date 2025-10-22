// Stage 4 exercise: add two large matrices using multiple blocks.

#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix.h"

#include "todo.cu"

constexpr int kMatrixSize = 500;
constexpr float kTolerance = 1e-5f;

void fillMatrix(Matrix matrix, float scale) {
    for (int row = 0; row < matrix.height; ++row) {
        for (int col = 0; col < matrix.width; ++col) {
            const int idx = row * matrix.width + col;
            matrix.elements[idx] = scale * static_cast<float>(idx);
        }
    }
}

#ifndef SKIP_CPU
void addMatricesHost(const Matrix& a, const Matrix& b, Matrix& out) {
    for (int row = 0; row < a.height; ++row) {
        for (int col = 0; col < a.width; ++col) {
            const int idx = row * a.width + col;
            out.elements[idx] = a.elements[idx] + b.elements[idx];
        }
    }
}

bool compareMatrices(const Matrix& lhs, const Matrix& rhs, float tolerance) {
    const std::size_t count = matrixElementCount(lhs);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::fabs(lhs.elements[i] - rhs.elements[i]) > tolerance) {
            std::cerr << "Mismatch at element " << i
                      << " lhs=" << lhs.elements[i]
                      << " rhs=" << rhs.elements[i] << '\n';
            return false;
        }
    }
    return true;
}
#endif

int runMatrixAddDemo() {
    std::vector<float> hostAData(kMatrixSize * kMatrixSize, 0.0f);
    std::vector<float> hostBData(kMatrixSize * kMatrixSize, 0.0f);
    std::vector<float> hostCData(kMatrixSize * kMatrixSize, 0.0f);
#ifndef SKIP_CPU
    std::vector<float> hostReferenceData(kMatrixSize * kMatrixSize, 0.0f);
#endif

    Matrix hostA{kMatrixSize, kMatrixSize, hostAData.data()};
    Matrix hostB{kMatrixSize, kMatrixSize, hostBData.data()};
    Matrix hostC{kMatrixSize, kMatrixSize, hostCData.data()};
#ifndef SKIP_CPU
    Matrix reference{kMatrixSize, kMatrixSize, hostReferenceData.data()};
#endif

    fillMatrix(hostA, 1.0f);
    fillMatrix(hostB, -0.5f);
#ifndef SKIP_CPU
    addMatricesHost(hostA, hostB, reference);
#endif

    CHECK_CUDA(addMatricesOnDevice(hostA, hostB, hostC));

#ifdef SKIP_CPU
    // Print all GPU results directly
    const std::size_t count = matrixElementCount(hostC);
    for (std::size_t i = 0; i < count; ++i) {
        std::cout << hostC.elements[i] << std::endl;
    }
#else
    if (compareMatrices(hostC, reference, kTolerance)) {
        std::cout << "Stage 4 matrix add matches reference ✅" << std::endl;
    } else {
        std::cout << "Stage 4 matrix add mismatch ❌" << std::endl;
    }
#endif

    return 0;
}

int main() {
    return runMatrixAddDemo();
}
