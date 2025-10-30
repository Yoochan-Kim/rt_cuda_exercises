#include <stdio.h>

/**
 * 2차원 블록에서 Warp 인덱싱
 * 2차원 인덱스를 1차원으로 변환한 후 계산
 */
__global__ void warpIndexing2D(int* output) {
    // 2차원 인덱스를 1차원으로 변환
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Warp 내에서의 위치 계산
    int laneId = threadId % 32;

    // 해당 스레드가 속한 Warp의 ID
    int warpId = threadId / 32;

    // 각 스레드의 정보를 저장
    // 출력 형식: [threadId] [blockDim.x] [blockDim.y] [warpId] [laneId]
    output[threadId * 5 + 0] = threadIdx.x;
    output[threadId * 5 + 1] = threadIdx.y;
    output[threadId * 5 + 2] = warpId;
    output[threadId * 5 + 3] = laneId;
    output[threadId * 5 + 4] = threadId;
}

int main() {
    int blockX = 32;  // 32개 스레드
    int blockY = 4;   // 4개 행 = 총 128개 스레드 = 4개 Warp
    int blockSize = blockX * blockY;
    int outputSize = blockSize * 5;

    // 호스트 메모리 할당
    int* h_output = (int*)malloc(outputSize * sizeof(int));

    // 디바이스 메모리 할당
    int* d_output;
    cudaMalloc(&d_output, outputSize * sizeof(int));

    // 커널 실행
    warpIndexing2D<<<1, dim3(blockX, blockY)>>>(d_output);

    // 결과 복사
    cudaMemcpy(h_output, d_output, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 결과 출력
    printf("X  | Y | Linear ID | Warp ID | Lane ID\n");
    printf("---+---+-----------+---------+--------\n");
    for (int y = 0; y < blockY; y++) {
        for (int x = 0; x < blockX; x++) {
            int idx = y * blockX + x;
            printf("%2d | %d |    %3d    |   %d     |   %2d\n",
                   h_output[idx*5+0], h_output[idx*5+1],
                   h_output[idx*5+4], h_output[idx*5+2], h_output[idx*5+3]);
        }
    }

    // 메모리 해제
    cudaFree(d_output);
    free(h_output);

    return 0;
}
