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

__global__ void scaleTile(const float* input, float* output, int width, int height) {
    __shared__ float tile[16][16];

    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalRow >= height || globalCol >= width) {
        return;
    }

    int index = globalRow * width + globalCol;

    tile[threadIdx.y][threadIdx.x] = input[index];
    __syncthreads();

    tile[threadIdx.y][threadIdx.x] *= 2.0f;
    __syncthreads();

    output[index] = tile[threadIdx.y][threadIdx.x];
}

int main() {
    const int width = 16;
    const int height = 16;
    const int count = width * height;
    const size_t bytes = count * sizeof(float);

    float hostInput[count];
    float hostOutput[count];
    for (int i = 0; i < count; ++i) {
        hostInput[i] = static_cast<float>(i);
    }

    float* devInput = nullptr;
    float* devOutput = nullptr;
    CHECK_CUDA(cudaMalloc(&devInput, bytes));
    CHECK_CUDA(cudaMalloc(&devOutput, bytes));
    CHECK_CUDA(cudaMemcpy(devInput, hostInput, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(width / block.x, height / block.y);
    scaleTile<<<grid, block>>>(devInput, devOutput, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hostOutput, devOutput, bytes, cudaMemcpyDeviceToHost));

    printf("hostOutput[0] = %.1f\n", hostOutput[0]);

    CHECK_CUDA(cudaFree(devInput));
    CHECK_CUDA(cudaFree(devOutput));
    return 0;
}
