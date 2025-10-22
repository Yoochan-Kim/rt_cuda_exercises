# Stage 03: 2D Thread Blocks

## 튜토리얼 목표

- 2차원 스레드 블록이 왜 필요한지 사례로 이해하기
- 스레드 좌표 `(row, col)`을 계산해 행렬 좌표로 매핑하기
- 스레드 수가 행렬 크기를 초과할 때 경계 검사 방법 이해하기

## 핵심 개념

### 1차원으로 행렬을 다루면 생기는 불편함

`Matrix` 구조체를 사용해 행렬을 복사(copy)한다고 가정해 봅시다. `Matrix`는 `width`, `height`, `elements` 포인터를 포함하며, CPU에서는 다음과 같이 2중 반복문을 작성해야 합니다:

```cpp
for (int row = 0; row < matrix.height; ++row) {
    for (int col = 0; col < matrix.width; ++col) {
        const int idx = row * matrix.width + col;
        dst.elements[idx] = src.elements[idx];
    }
}
```

1차원 스레드로 구현하면 `idx`를 다시 `row`, `col`로 분해하거나, 반복문을 한 번 더 돌려야 합니다. CUDA로 옮기면 다음과 같이 전역 인덱스를 기반으로 행과 열을 계산해야 합니다:

```cpp
__global__ void copyMatrix1D(const Matrix src, Matrix dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / src.width;
    int col = idx % src.width;
    dst.elements[idx] = src.elements[idx];
}
```

스레드가 직접 `(row, col)`을 계산해야 하므로 모듈로·나눗셈 연산이 추가되고, 코드가 복잡해집니다.

### 2차원 블록으로 풀어 보면

`dim3 block(x, y)`와 같이 블록에 두 축을 부여하면 스레드가 다음과 같이 생성됩니다.

- `threadIdx.x`: 열 방향 인덱스 (0 ~ x-1)
- `threadIdx.y`: 행 방향 인덱스 (0 ~ y-1)

스레드는 생성과 동시에 `row = threadIdx.y`, `col = threadIdx.x`를 얻을 수 있어 행렬 좌표를 자연스럽게 다룰 수 있습니다. 예를 들어 2차원 블록을 활용해 행렬을 복사하면 코드는 훨씬 간단해집니다:

```cpp
__global__ void copyMatrix2D(const Matrix src, Matrix dst) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    const int idx = row * src.width + col;
    dst.elements[idx] = src.elements[idx];
}
```

좌표를 직접 계산할 필요가 없고, 가독성도 좋아집니다.

### 경계 검사 패턴 이해하기

Stage 03부터는 스레드 개수와 행렬 요소 개수가 일치하지 않는 상황을 다룹니다. 예를 들어 `dim3 block(16, 16)` 안에는 256개의 스레드가 있지만, 8×8 행렬은 64개의 요소만 가지고 있습니다. 이때 범위를 벗어나는 스레드는 다음과 같이 조건문으로 건너뛰어야 합니다.

```cpp
if (row < src.height && col < src.width) {
    const int idx = row * src.width + col;
    dst.elements[idx] = src.elements[idx];
}
```

이 패턴은 이후 단계에서 여러 블록을 사용할 때도 동일하게 재사용합니다.

## 실습

### 목표

`dim3 block(kBlockDimX, kBlockDimY)` 구성으로 행렬 덧셈을 수행하는 CUDA 커널을 작성합니다.

### 단계

1. 커널에서 `int row = threadIdx.y;`, `int col = threadIdx.x;`를 계산합니다.
2. `if (row < height && col < width)` 조건으로 범위를 확인합니다.
3. 유효한 경우 `row * width + col` 위치에 두 입력 행렬의 합을 기록합니다.

### 기대 출력

```
Stage 3 matrix add matches reference ✅
```

### 참고

- 블록은 한 개(`dim3 grid(1, 1)`)만 사용하지만, 블록 크기(`kBlockDimX`, `kBlockDimY`)는 행렬보다 크게 설정되어 있습니다. 따라서 구현 시 반드시 경계 검사를 수행해야 합니다.
  ```cpp
  constexpr int kBlockDimX = 16;
  constexpr int kBlockDimY = 16;
  constexpr int kMatrixSize = 8;
  ```
- Stage 04에서는 여러 블록을 사용해 더 큰 행렬을 처리하는 방법을 배웁니다.
