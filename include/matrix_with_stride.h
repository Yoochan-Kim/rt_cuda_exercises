#pragma once

#include <cstddef>

// Matrix description with stride support for sub-matrix operations (stage06+).
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix {
    int width;
    int height;
    int stride;  // Row stride for sub-matrix support
    float* elements;
};

inline std::size_t matrixElementCount(const Matrix& matrix) {
    return static_cast<std::size_t>(matrix.width) *
           static_cast<std::size_t>(matrix.height);
}
