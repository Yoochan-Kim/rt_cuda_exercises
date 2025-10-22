// Stage 0 exercise: make both CPU and GPU print a greeting.
// Expected output (order matches host first, then 10 GPU threads):
// Stage 0: nvcc builds host and device code in one program.
// Hello from the CPU (host function)!
// Hello from the GPU (block 0, thread 0)!
// ...
// Hello from the GPU (block 0, thread 9)!

#include <iostream>
#include "cuda_utils.cuh"

void hostHello() {
    std::cout << "Hello from the CPU (host function)!" << std::endl;
}

__global__ void deviceHello() {
    printf("Hello from the GPU (block %d, thread %d)!\n", blockIdx.x, threadIdx.x);
}

int main_todo() {
    hostHello();

    /* TODO:
     * Launch the deviceHello kernel so that it runs exactly 10 threads.
     * For example: kernel<<<1, threadCount>>>();
     * Replace the entire line below with your kernel launch once you are ready.
     */

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
