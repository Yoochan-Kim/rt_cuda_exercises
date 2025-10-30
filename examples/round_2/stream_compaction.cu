#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef INNER
#define INNER 256
#endif

// ===== heavy work =====
__device__ float do_work(float acc, float t, int inner) {
    for (int k = 0; k < inner; ++k) {
        float a = acc + t * (0.0001f * (k + 1));
        float b = t   + acc * (0.0002f * (k + 3));
        float s = sinf(a);
        float c = cosf(b);
        acc = acc * (1.0f + 1e-6f) + s * 0.7f + c * 0.3f;
        t   = t   * (1.0f - 1e-6f) + s * 0.2f - c * 0.1f;
    }
    return acc;
}

// ===== Stream Compaction: phase 1 =====
__global__ void filterKernel(const float* input, int n,
                             float* filtered, int* indices,
                             int* count, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        if (v >= threshold) {
            int pos = atomicAdd(count, 1);
            filtered[pos] = v;
            indices[pos]  = idx;
        }
    }
}

// ===== Stream Compaction: phase 2 =====
__global__ void processFilteredKernel(const float* filtered, const int* indices,
                                      int filteredCount, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < filteredCount) {
        float acc = filtered[i];
        float t   = acc * 0.5f + (float)(indices[i]) * 1e-4f;
        float res = do_work(acc, t, INNER);
        output[indices[i]] = res;
    }
}

// ===== No-Compaction (branching, divergence) =====
__global__ void naiveProcessKernel(const float* input, int n,
                                   float* output, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        if (v >= threshold) {
            float t = v * 0.5f + (float)idx * 1e-4f;
            output[idx] = do_work(v, t, INNER);
        } else {
            output[idx] = -1.0f;
        }
    }
}

int main() {
    // ---- config ----
    const int n = 1 << 24;
    const float threshold = 50.0f;
    const int block = 256;
    const int gridAll  = (n + block - 1) / block;
    const int gridHalf = ((n + 1) / 2 + block - 1) / block;
    const int iters = 50;

    // ---- host buffers ----
    float *h_in  = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        h_in[i]  = (float)(i % 100);
        h_out[i] = -1.0f;
    }
    int h_count = 0;

    // ---- device buffers ----
    float *d_in = nullptr, *d_filtered = nullptr, *d_out = nullptr;
    int *d_indices = nullptr, *d_count = nullptr;
    cudaMalloc(&d_in,       n * sizeof(float));
    cudaMalloc(&d_filtered, n * sizeof(float));
    cudaMalloc(&d_indices,  n * sizeof(int));
    cudaMalloc(&d_out,      n * sizeof(float));
    cudaMalloc(&d_count,    sizeof(int));

    // ---- copy input once (kernel time만 비교) ----
    cudaMemcpy(d_in,  h_in,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, n * sizeof(float), cudaMemcpyHostToDevice);

    // ---- precompute filtered count once for fixed grid of phase2 ----
    cudaMemset(d_count, 0, sizeof(int));
    filterKernel<<<gridAll, block>>>(d_in, n, d_filtered, d_indices, d_count, threshold);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    const int gridProc = (h_count + block - 1) / block;

    // ---- warmup ----
    cudaMemset(d_out, 0xFF, n * sizeof(float));
    cudaMemset(d_count, 0, sizeof(int));
    filterKernel<<<gridAll, block>>>(d_in, n, d_filtered, d_indices, d_count, threshold);
    processFilteredKernel<<<gridProc, block>>>(d_filtered, d_indices, h_count, d_out);
    naiveProcessKernel<<<gridAll, block>>>(d_in, n, d_out, threshold);
    cudaDeviceSynchronize();

    // ---- timers ----
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // ---- measure: Stream Compaction (phase1+phase2) ----
    float ms_compact = 0.0f;
    cudaEventRecord(s);
    for (int it = 0; it < iters; ++it) {
        cudaMemset(d_out, 0xFF, n * sizeof(float));  // optional
        cudaMemset(d_count, 0, sizeof(int));
        filterKernel<<<gridAll, block>>>(d_in, n, d_filtered, d_indices, d_count, threshold);
        processFilteredKernel<<<gridProc, block>>>(d_filtered, d_indices, h_count, d_out);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms_compact, s, e);
    ms_compact /= iters;

    // ---- measure: No-Compaction (branching) ----
    float ms_naive = 0.0f;
    cudaEventRecord(s);
    for (int it = 0; it < iters; ++it) {
        cudaMemset(d_out, 0xFF, n * sizeof(float));  // optional
        naiveProcessKernel<<<gridAll, block>>>(d_in, n, d_out, threshold);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms_naive, s, e);
    ms_naive /= iters;

    // ---- results ----
    printf("n=%d, threshold=%.1f, INNER=%d, iters=%d\n", n, threshold, INNER, iters);
    printf("filtered_count=%d\n", h_count);
    printf("avg time (ms): compaction %.3f | no-compaction %.3f | speedup %.3fx\n",
           ms_compact, ms_naive, ms_naive / ms_compact);

    // ---- sample outputs (first 10 valid) ----
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    int shown = 0;
    for (int i = 0; i < n && shown < 10; ++i) {
        if (h_out[i] != -1.0f && isfinite(h_out[i])) {
            printf("out[%d]=%.6f\n", i, h_out[i]);
            ++shown;
        }
    }

    // ---- cleanup ----
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_in); cudaFree(d_filtered); cudaFree(d_indices); cudaFree(d_out); cudaFree(d_count);
    free(h_in); free(h_out);
    return 0;
}
