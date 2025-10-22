// Stage 2 exercise: implement the GPU-side vector addition kernel.

#include "cuda_utils.cuh"

/* TODO:
 * Each thread should add the corresponding elements of a and b.
 *   - Inputs: a, b (device arrays), c (device output), count (element count).
 *   - Steps:
 *     1) Compute the global index using blockIdx.x, blockDim.x, threadIdx.x.
 *     2) Store c[idx] = a[idx] + b[idx].
 */
__global__ void vectorAddKernel(const float* a,
                                const float* b,
                                float* c,
                                int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
