# Stage 09: Sequential Addressing Reduction

## 튜토리얼 목표

- Stage 08 커널이 여전히 겪는 공유 메모리 뱅크 컨플릭트를 이해한다.
- sequential addressing 루프를 구현해 뱅크 컨플릭트를 제거한다.

## Strided index의 남은 병목

Stage 08의 strided index 버전은 워프 다이버전스와 `%` 연산을 제거했지만, 여전히 공유 메모리 뱅크 컨플릭트가 남아 있습니다. stride가 커질수록 `sdata[index]`와 `sdata[index + stride]`가 같은 뱅크에 매핑되어 여러 스레드가 직렬화되고, 특히 초기 반복(iteration)에서는 한 번에 절반 이상의 스레드가 충돌을 일으킵니다.

공유 메모리 접근이 순차적일수록(인접한 스레드가 인접한 주소를 읽고 쓸수록) 컨플릭트 확률이 낮아집니다. 따라서 루프를 **sequential addressing** 방식으로 바꿔 인접 스레드가 인접 주소를 다루게 만들면, 뱅크 컨플릭트가 사라져 Stage 08 대비 성능이 꾸준히 향상됩니다.

## Sequential addressing 루프

Sequential addressing은 stride를 초기에 크게 잡았다가 절반씩 줄여 가며 앞쪽 절반의 스레드만 살아남는 방식입니다. 각 반복에서 활성 스레드는 연속된 공유 메모리 주소를 읽고 쓰므로, 워프 전체가 뱅크 충돌 없이 진행됩니다. 구현 관점에서는 다음 흐름을 따르면 됩니다.

1. `threadIdx.x`를 지역 변수에 저장해 반복문에서 재사용합니다.
2. 반복문 초깃값은 `blockDim.x / 2`, 이후 매 반복마다 `stride >>= 1`로 절반씩 줄입니다.
3. `tid < stride`인 스레드만 이웃 슬롯(`tid + stride`)을 더하고, 나머지는 대기합니다.
4. 각 반복 끝에는 `__syncthreads()`로 공유 메모리 업데이트를 동기화합니다.

- 입력 길이가 `blockDim.x`의 배수라는 가정 아래에서는 `tid + stride < blockDim.x` 조건이 자동으로 보장됩니다.
- Stage 08과 마찬가지로 공유 메모리에 초깃값을 적재한 뒤 루프가 끝나면 `tid == 0` 스레드가 블록 부분 합을 전역 메모리에 기록합니다.
- `stride`가 32 이하로 내려가도 warp-synchronous 동작을 의지하지 말고, 이후 단계에서 안전하게 언롤링을 적용할 예정입니다.

## 실습

### 목표

Stage 08 커널의 interleaved 루프를 sequential addressing 루프로 교체해 뱅크 컨플릭트를 제거한다.

### 단계

1. 공유 메모리 적재 및 `__syncthreads()` 호출 구조는 Stage 08과 동일하게 유지한다.
2. 반복문을 `for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)` 형태로 작성한다.
3. `tid < stride` 조건을 만족하는 스레드만 이웃 값을 더하고, 반복마다 `__syncthreads()`로 동기화한다.
4. 루프 종료 후 `tid == 0` 스레드가 `sdata[0]`을 `g_odata[blockIdx.x]`에 저장한다.

### 기대 출력

```
Stage 9 reduction matches reference ✅

Input size: 16777216 elements
CPU sum : <cpu_sum>
GPU sum : <gpu_sum>
Relative error: <error>

Timing:
  CPU time : <cpu_ms> ms
  GPU time : <gpu_ms> ms
```
