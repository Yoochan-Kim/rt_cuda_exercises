// Stage 11: Unroll the Last Warp
// Builds on first-add-during-load and removes the final warp's control-flow overhead with manual unrolling.

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 1024;
constexpr int kElementsPerThread = 2;

/* TODO:
 * Extend the first-add-during-load reduction by unrolling the last warp.
 * Stage 11 keeps the Stage 10 assumption that the input length is a multiple of blockDim.x * 2.
 * Steps:
 *   1) Declare shared memory with extern __shared__.
 *   2) Compute baseIdx = blockIdx.x * (blockDim.x * kElementsPerThread) + threadIdx.x.
 *   3) Load g_idata[baseIdx] and g_idata[baseIdx + blockDim.x], add them, and store the sum in shared memory.
 *   4) Run the sequential addressing loop while stride > 32, synchronizing after each iteration.
 *   5) For tid < 32, use a volatile pointer to shared memory and manually unroll the final six adds.
 *   6) Thread 0 writes sdata[0] to g_odata[blockIdx.x].
 */
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    // TODO: Manually unroll the last six additions using the volatile pointer.
}

__global__ void reduceSharedMemoryFirstAddWarpUnrollKernel(const float* g_idata,
                                                           float* g_odata,
                                                           unsigned int count) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    // TODO: Compute baseIdx = blockIdx.x * (blockDim.x * kElementsPerThread) + tid.
    // TODO: Load two elements, add them during the load phase, and store the sum in shared memory.
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        // TODO: Let threads with tid < stride accumulate the neighbor entry from shared memory.
        __syncthreads();
    }

    if (tid < 32) {
        // TODO: Call warpReduce with a volatile pointer to shared memory.
    }

    // TODO: Thread 0 writes the block's partial sum (sdata[0]) into g_odata[blockIdx.x].
}

// Host helper that prepares device buffers and collects per-block partial sums.
cudaError_t reduceSharedMemoryFirstAddWarpUnroll(const float* hostInput,
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

    // Launch the warp-unrolled reduction, synchronize, and copy the partial sums back into blockSums.
    const std::size_t sharedMemBytes = threads * sizeof(float);
    reduceSharedMemoryFirstAddWarpUnrollKernel<<<gridSize, threads, sharedMemBytes>>>(
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
