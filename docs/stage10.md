# Stage 10: First Add During Load

## 튜토리얼 목표

- Stage 09의 sequential addressing 루프가 여전히 겪는 “첫 반복에서 절반의 스레드가 놀고 있는” 비효율을 짚어 본다.
- 각 스레드가 **두 개의 요소를 읽어 즉시 더한 뒤** 공유 메모리에 저장하는 *first add during load* 패턴을 구현한다.
- 한 블록이 처리하는 입력 개수가 `blockDim.x` → `blockDim.x * 2`로 늘어난 상황에서도 기존 reduction 루프가 올바른 결과를 내는지 검증한다.

## 왜 첫 반복이 낭비일까?

Sequential addressing으로 바꿔도 루프의 첫 반복(`stride = blockDim.x / 2`)에서는 여전히 블록 스레드의 절반만 활성화됩니다. 이 반복은 “짝수/홀수” 슬롯을 합치는 역할을 할 뿐이며, 모든 스레드가 전역 메모리에서 단 한 개의 값만 읽어 왔기 때문에 **연산 수 대비 스레드 활용도가 낮습니다.** Stage 07~09와 마찬가지로 입력 길이가 블록 크기의 배수라고 가정할 수 있다면, 이 반복이 맡은 일을 **로드 시점**으로 끌어당기는 편이 더 효율적입니다.

이 문제는 전역 메모리에서 값을 가져올 때 이미 한 번의 덧셈을 수행해 두면 해결됩니다. 즉, 각 스레드가 `g_idata[i]`와 `g_idata[i + blockDim.x]`를 읽어 합쳐 놓으면, 루프의 첫 반복에서 할 일을 미리 수행한 셈이 됩니다.

## First add during load 패턴

핵심 변화는 **로드 단계에서 두 값을 읽어 더한 뒤 공유 메모리에 저장**하는 것입니다. 구현 단계는 다음과 같습니다.

1. 스레드 인덱스를 캐시합니다.

   ```cpp
   const unsigned int tid = threadIdx.x;
   const unsigned int global = blockIdx.x * (blockDim.x * 2) + tid;
   ```

2. 공유 메모리에 기록할 값을 계산합니다. Stage 10에서는 입력 길이가 `blockDim.x * 2`의 배수라고 가정하므로, 두 번째 읽기를 조건 없이 수행해도 됩니다.

   ```cpp
   const float first = g_idata[global];
   const float second = g_idata[global + blockDim.x];
   sdata[tid] = first + second;
   __syncthreads();
   ```

3. 공유 메모리에 모인 `blockDim.x`개의 부분합을 Stage 09와 동일한 sequential addressing 루프로 줄입니다.

   ```cpp
   for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
     if (tid < stride) {
       sdata[tid] += sdata[tid + stride];
     }
     __syncthreads();
   }
   ```

4. 스레드 0이 결과를 전역 메모리에 기록하는 마무리 절차는 동일합니다.

이 접근법은 전역 메모리 접근을 두 번으로 늘렸지만, 같은 워프가 연속 주소를 읽어 오므로 여전히 완전히 coalesced된 형태를 유지합니다. 또한 각 블록이 입력의 두 배를 처리하므로, 호스트 코드에서는

```
numBlocks = (elementCount + (blockDim * 2 - 1)) / (blockDim * 2);
```

처럼 그리드 크기를 조정해야 전체 입력을 빠짐없이 덮을 수 있습니다.

## 실습

### 목표

Stage 09 커널을 확장해 **first add during load**를 구현하고, 첫 반복의 유휴 스레드를 제거한다.

### 단계

1. 전역 인덱스 계산을 `blockIdx.x * (blockDim.x * 2) + threadIdx.x` 형태로 바꾸고, 각 스레드가 두 개의 연속 요소를 읽어 합친 뒤 공유 메모리에 저장한다.
2. 입력이 짝수 개 블록에 맞지 않을 수 있으므로 두 번째 읽기를 경계 조건으로 감싸 안전하게 처리한다.
3. 공유 메모리에 값이 준비되면 Stage 09에서 사용한 sequential addressing 루프를 그대로 실행해 블록 합을 계산한다.
4. 호스트 측에서는 한 블록이 처리하는 요소 수가 두 배가 되었음을 반영해 그리드 크기를 재계산한다.

### 기대 출력

```
Stage 10 reduction matches reference ✅

Input size: 16777216 elements
CPU sum : <cpu_sum>
GPU sum : <gpu_sum>
Relative error: <error>

Timing:
  CPU time : <cpu_ms> ms
  GPU time : <gpu_ms> ms
```

> 숫자는 실행 환경에 따라 달라집니다. 정확성 확인 후 Stage 07~09 결과와 실행 시간을 비교해, first add during load가 가져오는 개선폭을 체감해 보세요.
