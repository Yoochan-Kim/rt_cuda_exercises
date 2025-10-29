// Stage 11: Unroll the Last Warp
// Each thread still performs first-add-during-load, but the final warp work is handled by a
// warpReduce helper that takes a volatile pointer so loads are not cached across iterations.

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
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceSharedMemoryFirstAddWarpUnrollKernel(const float* g_idata,
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

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(reinterpret_cast<volatile float*>(sdata), tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
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
