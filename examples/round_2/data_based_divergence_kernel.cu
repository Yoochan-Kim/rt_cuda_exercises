#include <stdio.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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

__global__ void dataBasedDivergenceKernel(float* data, int n, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        if (x > 50.0f) {
            x = pathA(x, repeat);
        } else {
            x = pathB(x, repeat);
        }
        data[idx] = x;
    }
}

int main() {
    const int n = 1 << 22;
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int repeat = 512;      // 경로 복잡도 (필요시 조정)
    const int launches = 8;      // 커널 반복 실행 횟수

    float* h_data = (float*)malloc(n * sizeof(float));
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run = [&](int mode, float& ms) {
        if (mode == 0) { for (int i = 0; i < n; ++i) h_data[i] = 100.0f; }       // 모두 if 경로
        else if (mode == 1) { for (int i = 0; i < n; ++i) h_data[i] = 0.0f; }    // 모두 else 경로
        else { for (int i = 0; i < n; ++i) h_data[i] = (i & 1) ? 60.0f : 40.0f; }// 반반

        cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int k = 0; k < launches; ++k) {
            dataBasedDivergenceKernel<<<gridSize, blockSize>>>(d_data, n, repeat);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    };

    float t_high, t_low, t_half;
    run(0, t_high);
    run(1, t_low);
    run(2, t_half);

    printf("all-high: %.3f ms\n", t_high);
    printf("all-low: %.3f ms\n", t_low);
    printf("half-half: %.3f ms\n", t_half);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);
    return 0;
}
