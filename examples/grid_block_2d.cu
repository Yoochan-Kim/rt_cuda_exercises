#include <cuda_runtime.h>
#include <cstdio>

__global__ void annotateThread() {
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block(%d,%d) Thread(%d,%d) -> Coord(%d,%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           globalCol, globalRow);
}

int main() {
    dim3 grid(3, 2);
    dim3 block(5, 3);
    annotateThread<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}
