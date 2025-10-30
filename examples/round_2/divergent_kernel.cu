#include <stdio.h>
#include <cuda_runtime.h>

#ifndef REPEAT
#define REPEAT 8192
#endif

__global__ void divergentKernel(int* out, const int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int x = idx;
        if (flags[idx]) {
            for (int r = 0; r < REPEAT; ++r) {
                x = x * 1664525 + 1013904223;
                x ^= (x << 13);
                x = (x ^ (x >> 17)) * 2654435761u;
                x ^= (x >> 5);
            }
        } else {
            for (int r = 0; r < REPEAT; ++r) {
                x = x * 1103515245 + 12345;
                x ^= (x >> 11);
                x = (x ^ (x << 7)) * 2246822519u;
                x ^= (x << 3);
            }
        }
        out[idx] = x;
    }
}

int main() {
    const int n = 1 << 22;
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    int *d_out, *d_flags;
    int *h_flags = (int*)malloc(n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run = [&](int mode, float& ms) {
        if (mode == 0) { for (int i = 0; i < n; ++i) h_flags[i] = 1; }
        else if (mode == 1) { for (int i = 0; i < n; ++i) h_flags[i] = 0; }
        else { for (int i = 0; i < n; ++i) h_flags[i] = (i & 1); }
        cudaMemcpy(d_flags, h_flags, n * sizeof(int), cudaMemcpyHostToDevice);

        const int iters = 20;
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int k = 0; k < iters; ++k) {
            divergentKernel<<<gridSize, blockSize>>>(d_out, d_flags, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    };

    float t_if, t_else, t_half;
    run(0, t_if);
    run(1, t_else);
    run(2, t_half);

    printf("all-if: %.3f ms\n", t_if);
    printf("all-else: %.3f ms\n", t_else);
    printf("half-half: %.3f ms\n", t_half);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    cudaFree(d_flags);
    free(h_flags);
    return 0;
}
