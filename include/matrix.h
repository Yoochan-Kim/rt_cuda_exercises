#pragma once

#include <cstddef>

// Simple row-major matrix description used across stages.
// M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
    int width;
    int height;
    float* elements;
};

inline std::size_t matrixElementCount(const Matrix& matrix) {
    return static_cast<std::size_t>(matrix.width) *
           static_cast<std::size_t>(matrix.height);
}

