#include <stdio.h>
#include <cuda_runtime.h>

#ifndef REPEAT
#define REPEAT 4096
#endif

__device__ int workA(int x, int repeat) {
    for (int r = 0; r < repeat; ++r) {
        x = x * 1664525 + 1013904223;
        x ^= (x << 13);
        x = (x ^ (x >> 17)) * 2654435761u;
        x ^= (x >> 5);
    }
    return x;
}

__device__ int workB(int x, int repeat) {
    for (int r = 0; r < repeat; ++r) {
        x = x * 1103515245 + 12345;
        x ^= (x >> 11);
        x = (x ^ (x << 7)) * 2246822519u;
        x ^= (x << 3);
    }
    return x;
}

__global__ void indexBasedDivergenceKernel(int* out, const int* in, int n, int repeat, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int x = in[idx];
    int takeA = (mode == 0) ? 1 : (mode == 1) ? 0 : ((threadIdx.x & 16) == 0);
    if (takeA) x = workA(x, repeat); else x = workB(x, repeat);
    out[idx] = x;
}

int main() {
    const int n = 1 << 22;
    const int blockSize = 256;              // 32의 배수 유지
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int repeat = 512;
    const int launches = 8;

    int *h_in = (int*)malloc(n * sizeof(int));
    int *d_in, *d_out;
    for (int i = 0; i < n; ++i) h_in[i] = i ^ 0x5a5a5a5a;
    cudaMalloc(&d_in,  n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run = [&](int mode, float& ms) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int k = 0; k < launches; ++k) {
            indexBasedDivergenceKernel<<<gridSize, blockSize>>>(d_out, d_in, n, repeat, mode);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    };

    float t_allA, t_allB, t_half;
    run(0, t_allA);
    run(1, t_allB);
    run(2, t_half);

    printf("all-A: %.3f ms\n", t_allA);
    printf("all-B: %.3f ms\n", t_allB);
    printf("half-half: %.3f ms\n", t_half);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
