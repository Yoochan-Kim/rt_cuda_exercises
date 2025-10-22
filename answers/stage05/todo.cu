// Stage 5: Matrix Multiplication without Shared Memory
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE.

#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matrix.h"

#define BLOCK_SIZE 16

/* TODO:
 * Implement matrix multiplication kernel C = A × B.
 * Each thread computes one element of C by accumulating A[row][k] * B[k][col].
 * No boundary check needed (dimensions are multiples of BLOCK_SIZE).
 */
__global__ void matrixMulKernel(Matrix A, Matrix B, Matrix C) {
    float Cvalue = 0.0f;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < A.width; ++k) {
        Cvalue += A.elements[row * A.width + k] * B.elements[k * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}

cudaError_t multiplyMatricesOnDevice(const Matrix& hostA,
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
     * Launch the matrix multiplication kernel with a 2D grid.
     * Hint: Use simple division (/) instead of ceiling division since dimensions
     *       are multiples of BLOCK_SIZE.
     */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(hostB.width / dimBlock.x, hostA.height / dimBlock.y);
    matrixMulKernel<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC);

    // Check for errors
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
