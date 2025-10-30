#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

__device__ float do_work(float acc, float t, int inner) {
    for (int k = 0; k < inner; ++k) {
        float a = acc + t * (0.0001f * (k + 1));
        float b = t + acc * (0.0002f * (k + 3));
        float s = sinf(a);
        float c = cosf(b);
        acc = acc * (1.0f + 1e-6f) + s * 0.7f + c * 0.3f;
        t   = t   * (1.0f - 1e-6f) + s * 0.2f - c * 0.1f;
    }
    return acc;
}

__global__ void early_exit_kenel(float* out, int max_iters, int inner) {
    const int lane = threadIdx.x & 31;
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int limit = (lane < 16) ? 1 : max_iters;
    float acc = 0.f, v = 1e-6f * (0.5f + idx);
    for (int i = 0; i < max_iters; ++i) {
        if (i >= limit) break;
        float t = v + i * 1e-5f;
        acc = do_work(acc, t, inner);
    }
    out[idx] = acc;
}

__global__ void no_early_exit_kenel(float* out, int max_iters, int inner, int warps_short) {
    const int warp_id = blockIdx.x;
    const int idx     = warp_id * blockDim.x + threadIdx.x;
    const int limit   = (warp_id < warps_short) ? 1 : max_iters;
    float acc = 0.f, v = 1e-6f * (0.5f + idx);
    for (int i = 0; i < max_iters; ++i) {
        if (i >= limit) break;
        float t = v + i * 1e-5f;
        acc = do_work(acc, t, inner);
    }
    out[idx] = acc;
}

int main() {
    int sm_count = 0; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int warp_size   = 32;
    const int warps_total = sm_count * 4096;
    const int threads     = warp_size;
    const int blocks      = warps_total;

    const int max_iters   = 8192;
    const int inner       = 64;
    const int warps_short = warps_total / 2;
    const size_t N        = (size_t)blocks * threads;

    float* d_out = nullptr;
    cudaMalloc(&d_out, N * sizeof(float));

    cudaEvent_t s1,e1,s2,e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);

    early_exit_kenel<<<blocks, threads>>>(d_out, max_iters, inner);
    no_early_exit_kenel<<<blocks, threads>>>(d_out, max_iters, inner, warps_short);
    cudaDeviceSynchronize();

    cudaEventRecord(s1);
    early_exit_kenel<<<blocks, threads>>>(d_out, max_iters, inner);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms1 = 0.f; cudaEventElapsedTime(&ms1, s1, e1);

    cudaEventRecord(s2);
    no_early_exit_kenel<<<blocks, threads>>>(d_out, max_iters, inner, warps_short);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    float ms2 = 0.f; cudaEventElapsedTime(&ms2, s2, e2);

    printf("%.3f\n", ms1);
    printf("%.3f\n", ms2);

    cudaEventDestroy(s1); cudaEventDestroy(e1);
    cudaEventDestroy(s2); cudaEventDestroy(e2);
    cudaFree(d_out);
    return 0;
}
