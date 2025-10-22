// Stage 6: Matrix Multiplication with Shared Memory
// Uses tiled algorithm with shared memory to reduce global memory bandwidth.

#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matrix_with_stride.h"

#define BLOCK_SIZE 16

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

/* TODO:
 * Implement tiled matrix multiplication using shared memory.
 * Use GetSubMatrix to get tiles, load them into __shared__ arrays,
 * and synchronize with __syncthreads() before/after tile computation.
 */
__global__ void matrixMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0.0f;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // TODO: Declare shared memory for As and Bs tiles
        // TODO: Load tiles from global to shared memory
        // TODO: Synchronize before computation
        // TODO: Compute partial result and accumulate to Cvalue
        // TODO: Synchronize after computation
    }

    SetElement(Csub, row, col, Cvalue);
}

cudaError_t multiplyMatricesOnDevice(const Matrix& hostA,
                                     const Matrix& hostB,
                                     Matrix& hostC) {
    // Load A and B to device memory
    Matrix deviceA;
    deviceA.width = deviceA.stride = hostA.width;
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
    deviceB.width = deviceB.stride = hostB.width;
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
    deviceC.width = deviceC.stride = hostC.width;
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
     * Dimensions are multiples of BLOCK_SIZE, use simple division.
     */

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
