#include <cuda_runtime.h>
#include <cstdio>

__global__ void addNumbers(float a, float b) {
    float result = a + b;
    printf("Sum on GPU = %.1f\n", result);
}

int main() {
    addNumbers<<<1, 1>>>(3.5f, 4.5f);
    cudaDeviceSynchronize();
    return 0;
}
