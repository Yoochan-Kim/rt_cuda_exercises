// Stage 9: Sequential Addressing Reduction
// Implements the sequential addressing variant by reversing the loop and using threadID-based indexing.

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 1024;

/* TODO:
 * Implement the shared memory reduction using sequential addressing to avoid bank conflicts.
 * Steps:
 *   1) Declare a shared memory buffer sized to blockDim.x (use extern __shared__).
 *   2) Load one element per thread from global memory if the global index is in range, otherwise store 0.
 *   3) For stride = blockDim.x / 2; stride > 0; stride >>= 1, let tid < stride accumulate sdata[tid + stride].
 *   4) After the loop, thread 0 writes the block's partial sum (sdata[0]) into g_odata[blockIdx.x].
 * Remember to keep __syncthreads() so that shared memory updates are visible before the next step.
 */
__global__ void reduceSharedMemoryKernel(const float* g_idata,
                                         float* g_odata,
                                         unsigned int count) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int globalIdx = blockIdx.x * blockDim.x + tid;

    // TODO: Load g_idata[globalIdx] into shared memory (store 0.0f if the index is out of range).
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // TODO: Let only the threads with tid < stride accumulate their neighbor from shared memory.
        __syncthreads();
    }

    if (tid == 0) {
        // TODO: Write the block's partial sum (sdata[0]) to g_odata[blockIdx.x].
    }
}

// Host helper that prepares device buffers and collects per-block partial sums.
cudaError_t reduceSharedMemorySequentialAddressing(const float* hostInput,
                                                   std::size_t count,
                                                   float* outSum) {
    if (count == 0) {
        *outSum = 0.0f;
        return cudaSuccess;
    }

    const std::size_t inputBytes = count * sizeof(float);
    float* deviceData = nullptr;
    cudaError_t status = cudaMalloc(&deviceData, inputBytes);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(deviceData, hostInput, inputBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(deviceData);
        return status;
    }

    const unsigned int threads = kThreadsPerBlock;
    const unsigned int gridSize =
        static_cast<unsigned int>((count + threads - 1) / threads);

    float* devicePartials = nullptr;
    status = cudaMalloc(&devicePartials, gridSize * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(deviceData);
        return status;
    }

    std::vector<float> blockSums(gridSize, 0.0f);

    // Launch the sequential addressing reduction, synchronize, and copy the partial sums back into blockSums.
    const std::size_t sharedMemBytes = threads * sizeof(float);
    reduceSharedMemoryKernel<<<gridSize, threads, sharedMemBytes>>>(
        deviceData, devicePartials, static_cast<unsigned int>(count));
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(blockSums.data(),
                            devicePartials,
                            gridSize * sizeof(float),
                            cudaMemcpyDeviceToHost);
    }

    const cudaError_t kernelStatus = status;

    cudaFree(devicePartials);
    cudaFree(deviceData);

    if (kernelStatus != cudaSuccess) {
        return kernelStatus;
    }

    double finalSum = 0.0;
    for (float partial : blockSums) {
        finalSum += static_cast<double>(partial);
    }

    *outSum = static_cast<float>(finalSum);
    return cudaSuccess;
}
