#include <cuda_runtime.h>
#include <cstdio>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    helloKernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
