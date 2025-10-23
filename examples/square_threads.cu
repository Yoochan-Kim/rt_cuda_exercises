#include <cuda_runtime.h>
#include <cstdio>

__global__ void printSquare() {
    int tid = threadIdx.x;
    int square = tid * tid;
    printf("Thread %d squared = %d\n", tid, square);
}

int main() {
    dim3 grid(1);
    dim3 block(16);
    printSquare<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}
