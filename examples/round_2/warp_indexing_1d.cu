#include <stdio.h>

/**
 * 1차원 블록에서 Warp 인덱싱
 * laneId: Warp 내에서의 스레드 위치 (0-31)
 * warpId: 해당 스레드가 속한 Warp의 ID
 */
__global__ void warpIndexing1D(int* output) {
    int laneId = threadIdx.x % 32;      // Warp 내에서의 위치 (0-31)
    int warpId = threadIdx.x / 32;      // Warp의 ID

    // 각 스레드의 인덱싱 정보를 저장
    // 출력 형식: [스레드ID] [warpId] [laneId]
    output[threadIdx.x * 3 + 0] = threadIdx.x;
    output[threadIdx.x * 3 + 1] = warpId;
    output[threadIdx.x * 3 + 2] = laneId;
}

int main() {
    int blockSize = 128;  // 128개 스레드 = 4개 Warp
    int outputSize = blockSize * 3;

    // 호스트 메모리 할당
    int* h_output = (int*)malloc(outputSize * sizeof(int));

    // 디바이스 메모리 할당
    int* d_output;
    cudaMalloc(&d_output, outputSize * sizeof(int));

    // 커널 실행
    warpIndexing1D<<<1, blockSize>>>(d_output);

    // 결과 복사
    cudaMemcpy(h_output, d_output, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 결과 출력
    printf("Thread ID | Warp ID | Lane ID\n");
    printf("----------+---------+--------\n");
    for (int i = 0; i < blockSize; i++) {
        printf("   %3d    |   %d     |   %2d\n",
               h_output[i*3+0], h_output[i*3+1], h_output[i*3+2]);
    }

    // 메모리 해제
    cudaFree(d_output);
    free(h_output);

    return 0;
}
