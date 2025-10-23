#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void fillSquare(int* out, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        out[tid] = tid * tid;
    }
}

int main() {
    const int count = 16;
    const size_t bytes = count * sizeof(int);

    int hostResult[count] = {0};
    int* deviceResult = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceResult, bytes));

    dim3 grid(1);
    dim3 block(16);
    fillSquare<<<grid, block>>>(deviceResult, count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hostResult, deviceResult, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < count; ++i) {
        printf("hostResult[%d] = %d\n", i, hostResult[i]);
    }

    CHECK_CUDA(cudaFree(deviceResult));
    return 0;
}
