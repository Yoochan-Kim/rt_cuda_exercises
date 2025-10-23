#include <cuda_runtime.h>

__global__ void kernelSkeleton() {
    // GPU 전용 로직
}

int main() {
    kernelSkeleton<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
