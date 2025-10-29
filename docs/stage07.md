# Stage 07: Shared Memory Reduction Baseline

## 튜토리얼 목표

- 1차원 합계(reduction) 문제를 정의하고 입력 생성, CPU 참조 구현, 결과 검증 루틴을 준비한다.
- 공유 메모리로 데이터를 적재한 뒤 interleaved addressing 방식으로 reduction을 수행하는 기본 커널을 구현한다.

## Reduction이란?

Reduction은 **모든 요소를 하나의 값으로 접어(summarize) 나가는** 패턴입니다. 합계, 최대값, 논리 AND 등 다양한 집계 연산이 동일한 구조를 공유하기 때문에 GPU 최적화의 대표적인 예제로 다뤄집니다. 이번 실습에서는 `float` 배열의 합계(sum)를 다룹니다.

Stage 07은 reduction의 **가장 첫 번째 공유 메모리 버전**을 구현합니다. 이 단계에서 확보한 정확도/성능을 이후 스테이지에서 지속적으로 비교하게 됩니다.

## 공유 메모리 기반 나이브 reduction 커널

NVIDIA PDF의 `reduce0` 예제처럼, 한 블록에 들어온 데이터를 먼저 공유 메모리로 적재한 뒤 interleaved addressing 패턴으로 합쳐 나갑니다. 전역 메모리를 여러 번 읽고 쓰지 않기 때문에 Stage 07이 기존 CPU 대비 GPU 성능을 확인하기 위한 기준점이 됩니다.

- `extern __shared__` 또는 `__shared__` 배열을 사용해 블록 크기만큼의 버퍼를 확보합니다.
- 각 스레드는 `unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;` 위치에서 하나의 값을 읽어와 공유 메모리에 저장합니다. (입력 길이가 블록 크기의 배수라고 가정하므로, 범위를 벗어나는 스레드는 등장하지 않습니다.)
- 이후 stride를 1, 2, 4… 배수로 늘려 가면서 `(threadIdx.x % (2 * stride)) == 0` 조건을 만족하는 스레드만 이웃 값을 더합니다. 이 접근법이 바로 interleaved addressing이며, 단계가 진행될수록 활성 스레드 수가 절반씩 줄어듭니다.
- 각 반복 사이에는 공유 메모리 업데이트가 끝났는지 보장하기 위해 `__syncthreads()`를 호출합니다.
- 루프가 끝나면 `threadIdx.x == 0`인 스레드가 `sdata[0]`을 읽어 블록별 합계를 `g_odata[blockIdx.x]`에 기록합니다.

핵심 흐름은 다음 세 단계로 요약됩니다.

1. **공유 메모리 적재**: 전역 인덱스가 범위 안에 있는 스레드는 입력 값을 공유 메모리 슬롯에 저장합니다.
2. **interleaved reduction**: stride를 1 → 2 → 4 …로 키우며 `stride` 배수 스레드만 이웃 슬롯 값을 더합니다. 각 반복 사이에는 `__syncthreads()`로 동기화합니다.
3. **블록 결과 기록**: 스레드 0이 `sdata[0]`을 읽어 블록별 부분 합을 전역 메모리에 씁니다.

공유 메모리 크기는 `blockDim.x * sizeof(float)`이면 충분하며, 입력 길이가 블록 크기의 배수라는 전제하에 추가적인 경계 검사가 필요 없습니다.

이 버전은 전역 메모리를 반복적으로 읽지 않지만, stride가 커질수록 살아남는 스레드가 줄어드는 워프 다이버전스 문제는 여전히 존재합니다. Stage 08에서는 주소 계산 방식을 바꿔(Sequential Addressing) 이 문제를 완화하고, 이후 단계에서 추가적인 최적화를 적용하게 됩니다.

## 실습

### 목표

`answers/stage07/todo.cu`의 템플릿을 완성해 전역 메모리 기반 reduction을 구현하고, CPU와의 오차 및 실행 시간을 출력합니다.

### 단계

1. `std::mt19937`를 고정 시드(12345)로 초기화해 입력 벡터를 생성하고 CPU 참조 합계를 계산합니다.
2. 디바이스 메모리를 할당하고 입력 데이터를 복사합니다(입력 길이가 `blockDim.x`의 배수인지 확인).
3. 공유 메모리 기반 reduction 커널을 작성해 각 블록의 부분 합을 계산합니다.
4. 블록별 결과를 호스트로 복사해 최종 합계를 얻고, CPU 결과와 비교합니다.
5. CUDA 이벤트(또는 제공된 타이머)로 GPU 실행 시간을 기록해 이후 단계와 비교합니다.

### 기대 출력

```
Stage 7 reduction matches reference ✅

Input size: 16777216 elements
CPU sum : <cpu_sum>
GPU sum : <gpu_sum>
Relative error: <error>

Timing:
  CPU time : <cpu_ms> ms
  GPU time : <gpu_ms> ms
```

> 숫자는 실행 환경에 따라 달라지며, GPU 시간이 CPU보다 빠르지 않을 수도 있습니다. 중요한 것은 정확성을 확보하고 baseline을 기록해 두는 것입니다.

### 다음 단계 예고

- Stage 08에서는 interleaved addressing이 만드는 워프 다이버전스를 줄이기 위해 sequential addressing 패턴을 도입합니다.
- 이후 단계에서는 초기 load 합치기, 루프 언롤링, warp 단위 최적화를 차례로 적용해 성능을 끌어올립니다.

> Stage 07 이후 단계에서도 동일한 입력 데이터를 재사용할 수 있도록, 모든 템플릿에서 `std::mt19937`를 고정 시드(12345)로 사용합니다.
