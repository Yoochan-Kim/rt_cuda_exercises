// Stage 4 exercise: add two large matrices using multiple blocks.
// Matrix dimensions may not be multiples of BLOCK_SIZE, so keep boundary checks.

#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matrix.h"

#define BLOCK_SIZE 16

/* TODO:
 * Each thread should add the corresponding elements of A and B.
 * Unlike Stage 3, now we use multiple blocks to handle large matrices.
 * Steps:
 *   1) row = blockIdx.y * blockDim.y + threadIdx.y;
 *   2) col = blockIdx.x * blockDim.x + threadIdx.x;
 *   3) Store C.elements[row * C.width + col] = A.elements[...] + B.elements[...]
 *
 * Note: Because matrix dimensions may not align with BLOCK_SIZE, keep the boundary check.
 */
__global__ void addMatricesKernel(Matrix A, Matrix B, Matrix C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C.height && col < C.width) {
        const int idx = row * C.width + col;
        C.elements[idx] = A.elements[idx] + B.elements[idx];
    }
}

cudaError_t addMatricesOnDevice(const Matrix& hostA,
                                const Matrix& hostB,
                                Matrix& hostC) {
    // Load A and B to device memory
    Matrix deviceA;
    deviceA.width = hostA.width;
    deviceA.height = hostA.height;
    size_t size = hostA.width * hostA.height * sizeof(float);
    cudaError_t status = cudaMalloc(&deviceA.elements, size);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemcpy(deviceA.elements, hostA.elements, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceA.elements);
        return status;
    }

    Matrix deviceB;
    deviceB.width = hostB.width;
    deviceB.height = hostB.height;
    size = hostB.width * hostB.height * sizeof(float);
    status = cudaMalloc(&deviceB.elements, size);
    if (status != cudaSuccess) {
        cudaFree(deviceA.elements);
        return status;
    }
    status = cudaMemcpy(deviceB.elements, hostB.elements, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceB.elements);
        cudaFree(deviceA.elements);
        return status;
    }

    // Allocate C in device memory
    Matrix deviceC;
    deviceC.width = hostC.width;
    deviceC.height = hostC.height;
    size = hostC.width * hostC.height * sizeof(float);
    status = cudaMalloc(&deviceC.elements, size);
    if (status != cudaSuccess) {
        cudaFree(deviceB.elements);
        cudaFree(deviceA.elements);
        return status;
    }

    /* TODO:
     * Launch the kernel with a 2D grid.
     *
     * Grid size calculation:
     *   - Each block covers BLOCK_SIZE x BLOCK_SIZE elements
     *   - We need (matrix_size / BLOCK_SIZE) blocks per dimension, rounded up
     *   - Ceiling division: (size + BLOCK_SIZE - 1) / BLOCK_SIZE
     *
     * Steps:
     *   1) Calculate gridX and gridY using the formula above
     *   2) Create dim3 grid(gridX, gridY)
     *   3) Launch the kernel with <<<grid, block>>>
     */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((hostA.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (hostA.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    addMatricesKernel<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC);

    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status != cudaSuccess) {
        cudaFree(deviceC.elements);
        cudaFree(deviceB.elements);
        cudaFree(deviceA.elements);
        return status;
    }

    // Read C from device memory
    status = cudaMemcpy(hostC.elements, deviceC.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceC.elements);
    cudaFree(deviceB.elements);
    cudaFree(deviceA.elements);
    return status;
}
