# Stage 04: 2D Grid Tiling

## 튜토리얼 목표

- 2차원 블록을 여러 개 배치해 행렬 전체를 덮는 방법 이해하기
- 블록 크기(`BLOCK_SIZE`)와 그리드 크기를 분리해 설정하는 이유 알기
- `ceil` 패턴을 활용해 필요한 블록 개수를 계산하고 경계 검사를 적용하기

## 핵심 개념

### Stage 03과의 차이: 블록을 격자처럼 배치하기

Stage 03에서는 블록을 하나만 사용해 2차원 스레드를 소개했습니다. Stage 04에서는 동일한 2차원 블록을 여러 개 만들어 행렬 전체를 타일링합니다. `dim3 grid(gridX, gridY)`로 가로·세로 방향 블록 개수를 지정하며, 각 블록은 `dim3 block(BLOCK_SIZE, BLOCK_SIZE)` 크기만큼의 서브 영역을 담당합니다.

### `BLOCK_SIZE`를 고정하는 이유

- **코드 재사용**: 블록 크기를 고정하면 다양한 행렬 크기에 대해 동일한 커널을 재사용할 수 있습니다.
- **하드웨어 친화성**: GPU는 32개 스레드(워프) 단위로 명령을 발행하므로, 블록당 스레드 수를 32의 정수 배수(예: 16×16 = 256)로 맞추면 하드웨어가 스레드를 효율적으로 스케줄링할 수 있습니다.
- **타일 기반 사고**: 행렬을 BLOCK_SIZE × BLOCK_SIZE 타일로 나누어 생각하면 이후 스테이지에서 공유 메모리 최적화를 도입하기 쉬워집니다.

### 그리드 크기 계산: `ceil` 패턴

행렬 크기가 BLOCK_SIZE의 배수가 아닌 경우 마지막 블록이 온전히 채워지지 않으므로 나머지 요소를 처리할 추가 블록이 필요합니다. 이를 위해 각 방향의 블록 수를 올림으로 계산합니다:

```cpp
int gridX = (width  + BLOCK_SIZE - 1) / BLOCK_SIZE;
int gridY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
dim3 dimGrid(gridX, gridY);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
```

이렇게 하면 남는 요소가 있을 때도 필요 블록이 하나 더 배치되어 전체 영역을 안전하게 덮을 수 있습니다.

### 2차원 전역 좌표 계산

Stage 02에서 1차원 전역 인덱스를 계산했던 공식(`blockIdx.x * blockDim.x + threadIdx.x`)을 2차원으로 확장하면 다음과 같이 행과 열을 얻을 수 있습니다:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

이 공식을 사용하면 모든 블록과 스레드가 서로 겹치지 않도록 고유한 `(row, col)`을 얻게 됩니다.

### 경계 검사 유지하기

블록 수를 올림으로 계산하면 마지막 블록에는 행이나 열이 행렬 범위를 벗어나는 스레드가 대부분 포함됩니다. 
예를 들어 `width = 500`, `BLOCK_SIZE = 16`이면 `gridX = (500 + 15) / 16 = 32`가 되고, 마지막 블록의 좌표는 31×16 = 496부터 시작합니다. 이 블록 안에 있는 `col = 496~511` 스레드들은 모두 경계 밖에 접근하게 됩니다. 
따라서 Stage 02에서 사용한 패턴을 그대로 적용해 이러한 스레드들이 작업을 건너뛰도록 합니다:

```cpp
if (row < C.height && col < C.width) {
    int idx = row * C.width + col;
    C.elements[idx] = A.elements[idx] + B.elements[idx];
}
```

이 조건은 대형 행렬을 다룰 때도 커널의 안전성을 보장합니다.

## 실습

### 목표

`BLOCK_SIZE × BLOCK_SIZE` 블록을 여러 개 배치해 행렬 덧셈을 수행하는 CUDA 커널을 완성합니다.

### 단계

1. `dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);`를 정의합니다.
2. `gridX`, `gridY`를 `ceil` 패턴으로 계산해 `dim3 dimGrid(gridX, gridY);`를 생성합니다.
3. 커널 내부에서 `row`, `col`을 전역 좌표 공식으로 계산합니다.
4. `if (row < height && col < width)` 조건으로 경계를 확인한 뒤, 두 행렬의 합을 결과 행렬에 기록합니다.

### 기대 출력

```
Stage 4 matrix add matches reference ✅
```

### 참고

- Stage 04 답안에서는 다음과 같이 상수를 정의합니다:
  ```cpp
  constexpr int kMatrixSize = 500;  // 입력 행렬 크기 (BLOCK_SIZE로 나누어 떨어지지 않음)
  #define BLOCK_SIZE 16             // 블록 크기 (타일 크기)
  ```
- 입력 크기가 BLOCK_SIZE로 나누어 떨어지지 않더라도 커널이 안전하게 동작하도록 항상 경계 검사를 유지해 주세요.
