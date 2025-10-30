#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <random>

#define N (1024 * 1024 * 10)
#define BLOCK_SIZE 256
#define NUM_RUNS 20

inline void checkCuda(cudaError_t e){ if(e!=cudaSuccess){ fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); exit(1);} }

__global__ void k_coalesced(const int* __restrict__ in, int* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = in[idx] * 2;
}

__global__ void k_strided_read(const int* __restrict__ in, int* __restrict__ out, int n, int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int j = (idx * stride) % n;
        out[idx] = in[j] * 2;
    }
}

__global__ void k_indirect_read(const int* __restrict__ in, const int* __restrict__ ind, int* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int j = ind[idx];
        out[idx] = in[j] * 2;
    }
}

__global__ void k_trash(int* buf, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) buf[i] ^= 1;
}

static inline void trash_cache(int* d_dummy, int m){
    int grid = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_trash<<<grid, BLOCK_SIZE>>>(d_dummy, m);
    checkCuda(cudaDeviceSynchronize());
}

static inline float run_coalesced(const int* d_in, int* d_out, int n){
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t s,t; checkCuda(cudaEventCreate(&s)); checkCuda(cudaEventCreate(&t));
    checkCuda(cudaEventRecord(s));
    k_coalesced<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);
    checkCuda(cudaEventRecord(t));
    checkCuda(cudaEventSynchronize(t));
    float ms=0; checkCuda(cudaEventElapsedTime(&ms, s, t));
    checkCuda(cudaEventDestroy(s)); checkCuda(cudaEventDestroy(t));
    return ms;
}

static inline float run_strided(const int* d_in, int* d_out, int n, int stride){
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t s,t; checkCuda(cudaEventCreate(&s)); checkCuda(cudaEventCreate(&t));
    checkCuda(cudaEventRecord(s));
    k_strided_read<<<grid, BLOCK_SIZE>>>(d_in, d_out, n, stride);
    checkCuda(cudaEventRecord(t));
    checkCuda(cudaEventSynchronize(t));
    float ms=0; checkCuda(cudaEventElapsedTime(&ms, s, t));
    checkCuda(cudaEventDestroy(s)); checkCuda(cudaEventDestroy(t));
    return ms;
}

static inline float run_indirect(const int* d_in, const int* d_ind, int* d_out, int n){
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t s,t; checkCuda(cudaEventCreate(&s)); checkCuda(cudaEventCreate(&t));
    checkCuda(cudaEventRecord(s));
    k_indirect_read<<<grid, BLOCK_SIZE>>>(d_in, d_ind, d_out, n);
    checkCuda(cudaEventRecord(t));
    checkCuda(cudaEventSynchronize(t));
    float ms=0; checkCuda(cudaEventElapsedTime(&ms, s, t));
    checkCuda(cudaEventDestroy(s)); checkCuda(cudaEventDestroy(t));
    return ms;
}

int main(){
    const int n = N;
    const size_t bytes = (size_t)n * sizeof(int);

    std::vector<int> h_in(n);
    std::iota(h_in.begin(), h_in.end(), 0);

    std::vector<int> h_ind(n);
    std::iota(h_ind.begin(), h_ind.end(), 0);
    std::mt19937 rng(123);
    for(int i=n-1;i>0;--i){
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(h_ind[i], h_ind[j]);
    }

    int *d_in=nullptr, *d_out=nullptr, *d_ind=nullptr;
    checkCuda(cudaMalloc(&d_in, bytes));
    checkCuda(cudaMalloc(&d_out, bytes));
    checkCuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&d_ind, bytes));
    checkCuda(cudaMemcpy(d_ind, h_ind.data(), bytes, cudaMemcpyHostToDevice));

    const int m = n * 2;
    int *d_dummy=nullptr;
    checkCuda(cudaMalloc(&d_dummy, (size_t)m * sizeof(int)));
    checkCuda(cudaMemset(d_dummy, 0, (size_t)m * sizeof(int)));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_coalesced<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);
    checkCuda(cudaDeviceSynchronize());
    trash_cache(d_dummy, m);

    float sum_coal=0, sum_stride=0, sum_rand=0;
    for(int r=0;r<NUM_RUNS;r++){
        trash_cache(d_dummy, m);
        sum_coal += run_coalesced(d_in, d_out, n);

        trash_cache(d_dummy, m);
        sum_stride += run_strided(d_in, d_out, n, 4);

        trash_cache(d_dummy, m);
        sum_rand += run_indirect(d_in, d_ind, d_out, n);
    }

    printf("Coalesced %.4f ms\n", sum_coal/NUM_RUNS);
    printf("Strided %.4f ms\n",   sum_stride/NUM_RUNS);
    printf("Random  %.4f ms\n",   sum_rand/NUM_RUNS);

    cudaFree(d_dummy);
    cudaFree(d_ind);
    cudaFree(d_out);
    cudaFree(d_in);
    return 0;
}
