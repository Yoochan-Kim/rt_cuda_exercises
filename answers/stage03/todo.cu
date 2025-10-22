// Stage 3 exercise: add two matrices on the GPU using 2D thread indexing.

#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "matrix.h"

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

/* TODO:
 * Each thread should add the corresponding elements of a and b.
 * Steps:
 *   1) row = threadIdx.y; col = threadIdx.x;
 *   2) If row < height and col < width, store c[row * width + col] = a[...] + b[...].
 */
__global__ void addMatricesKernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int width,
                                  int height) {
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    if (row < height && col < width) {
        const int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

cudaError_t addMatricesOnDevice(const Matrix& hostA,
                                const Matrix& hostB,
                                Matrix& hostC) {
    const std::size_t elementCount = matrixElementCount(hostA);
    const std::size_t byteSize = elementCount * sizeof(float);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    cudaError_t status = cudaMalloc(&deviceA, byteSize);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMalloc(&deviceB, byteSize);
    if (status != cudaSuccess) {
        cudaFree(deviceA);
        return status;
    }
    status = cudaMalloc(&deviceC, byteSize);
    if (status != cudaSuccess) {
        cudaFree(deviceB);
        cudaFree(deviceA);
        return status;
    }

    status = cudaMemcpy(deviceA, hostA.elements, byteSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceC);
        cudaFree(deviceB);
        cudaFree(deviceA);
        return status;
    }
    status = cudaMemcpy(deviceB, hostB.elements, byteSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceC);
        cudaFree(deviceB);
        cudaFree(deviceA);
        return status;
    }

    /* TODO:
     * Configure the launch parameters and launch the kernel.
     * Hint: Use a single grid block (dim3 grid(1, 1)) and a 2D thread block
     *       whose dimensions are defined by kBlockDimX/kBlockDimY. Some threads
     *       will fall outside the matrix and rely on the boundary check.
     */
    const dim3 block(kBlockDimX, kBlockDimY);
    const dim3 grid(1, 1);
    addMatricesKernel<<<grid, block>>>(deviceA, deviceB, deviceC, hostA.width, hostA.height);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status != cudaSuccess) {
        cudaFree(deviceC);
        cudaFree(deviceB);
        cudaFree(deviceA);
        return status;
    }

    status = cudaMemcpy(hostC.elements, deviceC, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceC);
    cudaFree(deviceB);
    cudaFree(deviceA);
    return status;
}
