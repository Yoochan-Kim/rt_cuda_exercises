# Stage 05: Matrix Multiplication Basics

## 튜토리얼 목표

- 행렬 곱셈 공식을 다시 확인하고, 결과 행렬의 각 요소가 어떻게 계산되는지 이해하기
- Stage 04에서 익힌 2차원 블록/그리드 구성을 적용해 GPU에서 행렬 곱셈을 수행하는 방법 익히기
- 한 스레드가 결과 행렬의 한 요소를 계산하면서 내부 루프를 통해 곱셈-누적을 수행하는 흐름 파악하기

## 핵심 개념

### 행렬 곱셈

2×3 행렬과 3×2 행렬을 곱하면 2×2 행렬이 만들어집니다. 각 위치는 행과 열을 내적해서 얻습니다:

```
A = [ 1 2 3 ]    B = [ 7  8 ]    C = A × B = [  (1×7 + 2×9 + 3×11)   (1×8 + 2×10 + 3×12) ]
    [ 4 5 6 ]        [ 9 10 ]                [  (4×7 + 5×9 + 6×11)   (4×8 + 5×10 + 6×12) ]
                     [11 12 ]

     = [  58  64 ]
       [ 139 154 ]
```

### CPU에서의 행렬 곱셈 복습

행렬 곱셈은 다음 세 중첩 반복문으로 표현할 수 있습니다. `A`는 `M×K`, `B`는 `K×N`, 결과 `C`는 `M×N` 크기를 가집니다:

```cpp
for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row][k] * B[k][col];
        }
        C[row][col] = sum;
    }
}
```

결과 행렬 `C[row][col]`는 `A`의 `row`번째 행과 `B`의 `col`번째 열을 내적(dot product)한 값입니다.

이제 이 계산을 GPU로 옮겨 봅시다. 결과 행렬을 `BLOCK_SIZE × BLOCK_SIZE` 타일로 나누고, 각 타일을 2차원 스레드 블록이 담당합니다. 
각 스레드는 `(row, col)` 좌표를 계산해 자신의 결과 셀을 누적하여 결과값을 구합니다.
이 단계에서는 모든 차원이 `BLOCK_SIZE`의 배수이므로 경계 조건을 따로 확인할 필요가 없습니다.

## 실습

### 목표

2차원 그리드/블록을 구성해 결과 행렬의 각 요소를 누적하는 행렬 곱셈 커널을 완성합니다.

### 단계

1. `dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);`와 `dim3 dimGrid(gridX, gridY);`를 `ceil` 패턴으로 계산합니다.
2. 커널에서 전역 좌표 `(row, col)`을 계산합니다.
3. 반복문을 돌며 `sum += A[row][k] * B[k][col];`을 누적하고, 최종 값을 `C[row][col]`에 저장합니다.
4. 호스트 코드에서 CPU 버전과 비교해 결과가 일치하는지 확인하고, 제공된 타이머로 CPU/GPU 수행 시간을 살펴봅니다.

### 기대 출력

```
Stage 5 matrix multiplication matches reference ✅

Performance Comparison:
  CPU Time: <cpu-ms> ms
  GPU Time: <gpu-ms> ms
  Speedup: <cpu-ms/gpu-ms>x
```

> 시간 값은 환경에 따라 달라집니다.

### 참고

- Stage 05 답안에서는 다음과 같이 차원을 정의합니다:
  ```cpp
  constexpr int M = 5120;
  constexpr int K = 2048;
  constexpr int N = 3072;
  #define BLOCK_SIZE 16
  ```
- Stage 06에서는 메모리 재사용 전략을 통해 성능을 개선하는 방법을 학습하게 됩니다.
