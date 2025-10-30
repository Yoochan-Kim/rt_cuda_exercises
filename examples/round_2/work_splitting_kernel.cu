#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef REPEAT
#define REPEAT 64
#endif

// ====== 핵심 연산 경로 ======
__device__ float pathA(float x, int repeat) {
    for (int r = 0; r < repeat; ++r) {
        x = fmaf(x, 1.000123f, 0.314159f);
        x = sinf(x) + logf(fabsf(x) + 1.0f);
        x = fmaf(x, 0.999871f, 1.234567f);
        x = sqrtf(fabsf(x) + 1e-3f);
    }
    return x;
}

__device__ float pathB(float x, int repeat) {
    for (int r = 0; r < repeat; ++r) {
        x = fmaf(x, 0.999771f, 2.7182818f);
        x = cosf(x) + expf(-fabsf(x));
        x = fmaf(x, 1.000431f, 0.5772157f);
        x = rsqrtf(fabsf(x) + 1e-3f);
    }
    return x;
}

// ====== Work Splitting 커널 ======
__global__ void evenProcessKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int realIdx = idx * 2;
    if (realIdx < n) {
        data[realIdx] = pathA(data[realIdx], REPEAT);
    }
}

__global__ void oddProcessKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int realIdx = idx * 2 + 1;
    if (realIdx < n) {
        data[realIdx] = pathB(data[realIdx], REPEAT);
    }
}

// ====== 단일(분기) 커널 ======
__global__ void unifiedProcessKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if ((idx & 1) == 0) {
            data[idx] = pathA(data[idx], REPEAT);
        } else {
            data[idx] = pathB(data[idx], REPEAT);
        }
    }
}

int main() {
    const int n = 1 << 24;
    const int blockSize = 256;
    const int gridSizeHalf = ((n + 1) / 2 + blockSize - 1) / blockSize;
    const int gridSizeAll  = (n + blockSize - 1) / blockSize;

    // 호스트 메모리
    float* h_data = (float*)malloc(n * sizeof(float));
    float* h_init = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        h_data[i] = (float)((i % 100) - 50) * 0.1f;
        h_init[i] = h_data[i];
    }

    // 디바이스 메모리
    float* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(float));
    size_t bytes = n * sizeof(float);

    // 타이밍 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int iters = 100;

    // ===== Work Splitting 측정 =====
    float ms_split_total = 0.0f;
    cudaEventRecord(start);
    for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice);
        evenProcessKernel<<<gridSizeHalf, blockSize>>>(d_data, n);
        oddProcessKernel <<<gridSizeHalf, blockSize>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_split_total, start, stop);
    float ms_split_avg = ms_split_total / iters;

    // ===== Unified(분기) 측정 =====
    float ms_unified_total = 0.0f;
    cudaEventRecord(start);
    for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice);
        unifiedProcessKernel<<<gridSizeAll, blockSize>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_unified_total, start, stop);
    float ms_unified_avg = ms_unified_total / iters;

    // 결과 출력
    printf("Timing (avg over %d iters, n=%d, REPEAT=%d):\n", iters, n, REPEAT);
    printf("  Work-Splitting (even+odd): %.3f ms\n", ms_split_avg);
    printf("  Unified (branching)      : %.3f ms\n", ms_unified_avg);
    printf("  Speedup (Split vs Unified): %.3fx\n", ms_unified_avg / ms_split_avg);

    // 검증(Work Splitting 결과)
    cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice);
    evenProcessKernel<<<gridSizeHalf, blockSize>>>(d_data, n);
    oddProcessKernel <<<gridSizeHalf, blockSize>>>(d_data, n);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("\nFirst 16 results:\n");
    for (int i = 0; i < 16; ++i) {
        printf("[%d]=%.6f%s", i, h_data[i], (i % 4 == 3) ? "\n" : "  ");
    }

    // 정리
    cudaFree(d_data);
    free(h_init);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
