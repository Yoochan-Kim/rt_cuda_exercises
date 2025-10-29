// Stage 10: First Add During Load
// Each thread pulls in two elements, adds them during the load phase, and then runs
// the sequential addressing loop on the pre-accumulated shared memory values.

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 1024;
constexpr int kElementsPerThread = 2;

/* TODO:
 * Implement the "first add during load" variant of the shared memory reduction.
 * Steps:
 *   1) Declare a shared memory buffer sized to blockDim.x (use extern __shared__).
 *   2) Compute the base index as blockIdx.x * (blockDim.x * 2) + threadIdx.x.
 *   3) Load two elements (base and base + blockDim.x), add them immediately, and store the sum in shared memory.
 *   4) Run the sequential addressing loop from Stage 09 on the shared memory buffer.
 *   5) Thread 0 writes the block's partial sum (sdata[0]) into g_odata[blockIdx.x].
 * Stage 10 assumes the input length is a multiple of blockDim.x * 2, so no boundary checks are required.
 */
__global__ void reduceSharedMemoryFirstAddKernel(const float* g_idata,
                                                 float* g_odata,
                                                 unsigned int count) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int baseIdx = blockIdx.x * (blockDim.x * kElementsPerThread) + tid;
    const unsigned int secondIdx = baseIdx + blockDim.x;

    const float first = g_idata[baseIdx];
    const float second = g_idata[secondIdx];
    sdata[tid] = first + second;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Host helper that prepares device buffers and collects per-block partial sums.
cudaError_t reduceSharedMemoryFirstAddDuringLoad(const float* hostInput,
                                                 std::size_t count,
                                                 float* outSum) {
    if (count == 0) {
        *outSum = 0.0f;
        return cudaSuccess;
    }

    const std::size_t elementsPerBlock =
        static_cast<std::size_t>(kThreadsPerBlock) * kElementsPerThread;
    if (count % elementsPerBlock != 0) {
        return cudaErrorInvalidValue;
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
        static_cast<unsigned int>(count / elementsPerBlock);

    float* devicePartials = nullptr;
    status = cudaMalloc(&devicePartials, gridSize * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(deviceData);
        return status;
    }

    std::vector<float> blockSums(gridSize, 0.0f);

    // Launch the first-add-during-load reduction, synchronize, and copy the partial sums back into blockSums.
    const std::size_t sharedMemBytes = threads * sizeof(float);
    reduceSharedMemoryFirstAddKernel<<<gridSize, threads, sharedMemBytes>>>(
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
