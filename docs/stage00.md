# Stage 00: Hello World

## 튜토리얼 목표

- CUDA 커널의 기본 문법 이해
- 여러 스레드 실행하기
- 블록과 스레드 인덱스 확인하기

## 핵심 개념

### CUDA 커널이란?

CUDA 커널은 GPU에서 실행되는 함수입니다. `__global__` 키워드로 정의합니다:

```cpp
__global__ void myKernel() {
    // 이 코드는 GPU에서 실행됩니다
}
```

### 커널 실행하기

커널은 호스트(CPU) 코드에서 `<<<...>>>` 문법으로 실행합니다:

```cpp
myKernel<<<gridSize, blockSize>>>();
```

- **첫 번째 인자 (gridSize)**: 블록의 개수
- **두 번째 인자 (blockSize)**: 블록당 스레드 개수

예를 들어, 1개 블록에 10개 스레드를 실행하려면:

```cpp
myKernel<<<1, 10>>>();
```

이렇게 하면 커널 함수가 **10번 병렬로** 실행됩니다!

### 스레드 식별하기

각 스레드는 고유한 ID를 가지며, 다음 변수로 확인할 수 있습니다:

- `blockIdx.x`: 현재 블록의 인덱스 (0부터 시작)
- `threadIdx.x`: 블록 내 스레드의 인덱스 (0부터 시작)

```cpp
__global__ void helloKernel() {
    printf("Hello from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}
```

위 커널을 `<<<1, 10>>>`으로 실행하면:
- 블록 0, 스레드 0
- 블록 0, 스레드 1
- ...
- 블록 0, 스레드 9

총 10개의 메시지가 출력됩니다.

### 동기화

커널 실행 후 `cudaDeviceSynchronize()`를 호출하여 모든 스레드가 완료될 때까지 기다립니다:

```cpp
myKernel<<<1, 10>>>();
cudaDeviceSynchronize();  // 모든 스레드 완료 대기
```

### 에러 체크

CUDA API 호출 후에는 항상 에러를 확인해야 합니다. `CHECK_CUDA` 매크로를 사용하면 편리합니다:

```cpp
#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
```

**사용 예시:**
```cpp
myKernel<<<1, 10>>>();
CHECK_CUDA(cudaGetLastError());        // 커널 실행 에러 체크
CHECK_CUDA(cudaDeviceSynchronize());   // 동기화 에러 체크
```

**중요**:
- `cudaGetLastError()`는 커널 실행 시 발생한 에러를 확인합니다
- 모든 CUDA API 호출 후 에러를 체크하는 것이 좋습니다

## 실습

### 목표

1개 블록, 10개 스레드를 실행하여 각 스레드가 자신의 블록/스레드 번호를 출력하도록 구성합니다. 블록 개수는 1, 스레드 개수는 10으로 맞춥니다.

### 기대 출력

```
Hello from the GPU (block 0, thread 0)!
Hello from the GPU (block 0, thread 1)!
...
Hello from the GPU (block 0, thread 9)!
```

### 참고

`<<<1, 10>>>`으로 실행하면 1개 블록, 10개 스레드가 생성됩니다.
