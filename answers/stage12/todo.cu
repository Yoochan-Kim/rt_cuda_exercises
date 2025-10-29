// Stage 12: Complete Unrolling with Templates
// Builds on Stage 11 by instantiating a fully unrolled kernel per block size using templates.

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 1024;
constexpr int kElementsPerThread = 2;

/* TODO:
 * Turn the Stage 11 implementation into a fully unrolled template-based kernel.
 * Steps:
 *   1) Make warpReduce a function template that conditionally performs the last six adds.
 *   2) Template the main kernel on blockSize and replace the stride loop with if (blockSize >= …) blocks.
 *   3) Launch the correct specialization from the host helper via a switch over the chosen block size.
 *   4) Preserve the first-add-during-load pattern and the assumption that count is a multiple of blockDim.x * 2.
 */
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) {
        sdata[tid] += sdata[tid + 32];
    }
    if (blockSize >= 32 && tid < 16) {
        sdata[tid] += sdata[tid + 16];
    }
    if (blockSize >= 16 && tid < 8) {
        sdata[tid] += sdata[tid + 8];
    }
    if (blockSize >= 8 && tid < 4) {
        sdata[tid] += sdata[tid + 4];
    }
    if (blockSize >= 4 && tid < 2) {
        sdata[tid] += sdata[tid + 2];
    }
    if (blockSize >= 2 && tid == 0) {
        sdata[tid] += sdata[tid + 1];
    }
}

template <unsigned int blockSize>
__global__ void reduceSharedMemoryFirstAddFullyUnrolledKernel(const float* g_idata,
                                                              float* g_odata,
                                                              unsigned int count) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int baseIdx = blockIdx.x * (blockSize * kElementsPerThread) + tid;
    const unsigned int secondIdx = baseIdx + blockSize;
    (void)count;

    const float first = g_idata[baseIdx];
    const float second = g_idata[secondIdx];
    sdata[tid] = first + second;
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<blockSize>(reinterpret_cast<volatile float*>(sdata), tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Host helper that prepares device buffers and collects per-block partial sums.
cudaError_t reduceSharedMemoryFirstAddFullyUnrolled(const float* hostInput,
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
    const std::size_t elementsPerBlock =
        static_cast<std::size_t>(threads) * kElementsPerThread;
    if (count % elementsPerBlock != 0) {
        cudaFree(deviceData);
        return cudaErrorInvalidValue;
    }

    const unsigned int gridSize =
        static_cast<unsigned int>(count / elementsPerBlock);

    float* devicePartials = nullptr;
    status = cudaMalloc(&devicePartials, gridSize * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(deviceData);
        return status;
    }

    std::vector<float> blockSums(gridSize, 0.0f);

    // Launch the fully unrolled reduction, synchronize, and copy the partial sums back into blockSums.
    const std::size_t sharedMemBytes = threads * sizeof(float);
    switch (threads) {
        case 1024:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<1024><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 512:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<512><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 256:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<256><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 128:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<128><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 64:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<64><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 32:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<32><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 16:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<16><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 8:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<8><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 4:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<4><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        case 2:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<2><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
        default:
            reduceSharedMemoryFirstAddFullyUnrolledKernel<1><<<gridSize, threads, sharedMemBytes>>>(
                deviceData, devicePartials, static_cast<unsigned int>(count));
            break;
    }
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
