// Stage 1 exercise: allocate device memory, run a kernel, copy results back.

#include <vector>
#include "cuda_utils.cuh"

/* TODO:
 * Write each thread's global index into deviceData.
 * - Goal: after completion deviceData contains 0..(count - 1).
 * - Steps:
 *   1) Compute idx = threadIdx.x (single block launch).
 *   2) Store idx into deviceData[idx].
 */
constexpr int kBlocks = 1;
constexpr int kThreads = 128;

__global__ void writeThreadIds(int* deviceData, int count) {
    const int idx = threadIdx.x;
    deviceData[idx] = idx;
}

bool runWriteThreadIdsTest(int elementCount, std::vector<int>& hostData) {
    const size_t bufferSize = elementCount * sizeof(int);

    int* deviceData = nullptr;
    /* TODO:
     * Allocate device memory large enough for elementCount ints.
     * Hint: CHECK_CUDA(cudaMalloc(<device pointer>, <byte size>));
     */
    CHECK_CUDA(cudaMalloc(&deviceData, bufferSize));

    /* TODO:
     * Launch writeThreadIds so that every element is written exactly once.
     * Replace the entire line below with your kernel launch once you are ready.
     * Hint: use kBlocks and kThreads for the <<<grid, block>>> configuration.
     */
    writeThreadIds<<<kBlocks, kThreads>>>(deviceData, elementCount);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* TODO:
     * Copy results back to host memory.
     * Hint: CHECK_CUDA(cudaMemcpy(hostData.data(), <device pointer>, <byte size>, cudaMemcpyDeviceToHost));
     */
    CHECK_CUDA(cudaMemcpy(hostData.data(), deviceData, bufferSize, cudaMemcpyDeviceToHost));

    /* TODO:
     * Free the device buffer once you are done using it.
     * Hint: CHECK_CUDA(cudaFree(<device pointer>));
     */
    CHECK_CUDA(cudaFree(deviceData));

    return true;
}
