# Stage 08: Strided Index Interleaved Reduction

## 튜토리얼 목표

- Stage 07에서 사용한 `tid % (2 * s) == 0` 조건이 만드는 워프 다이버전스와 `%` 연산 비용을 분석한다.
- **Strided index**를 사용해 활성 스레드를 정렬시키고, 모듈로 연산 없이 interleaved addressing을 수행한다.
- 호스트 파이프라인은 동일하게 유지하면서 커널 내부 루프만 교체해도 결과가 동일한지 검증한다.

## 왜 `%` 기반 interleaved 루프가 문제인가?

Stage 07 커널은 stride가 커질수록 `tid % (2 * s) == 0` 조건을 만족하는 스레드만 남습니다. 한 워프 안에서 실행 여부가 갈리기 때문에 다이버전스가 발생하고, 매 반복마다 `%` 연산을 수행해야 하는 오버헤드가 생깁니다. 비록 결과는 올바르더라도 GPU가 효율적으로 파이프라인을 채우지 못합니다.

## Strided index로 분기 정렬하기

`tid` 대신 `index = 2 * s * tid`를 사용하면, 각 반복에서 **앞쪽 절반 워프가 모두 살아남는** 형태로 재구성할 수 있습니다. 조건문은 단순한 범위 비교(`index < blockDim.x`)이며, `%` 연산 없이도 interleaved addressing을 유지합니다.

핵심 루프 구조는 다음과 같습니다.

```cpp
unsigned int tid = threadIdx.x;

for (unsigned int s = 1; s < blockDim.x; s *= 2) {
  unsigned int index = 2 * s * tid;
  if (index < blockDim.x) {
    sdata[index] += sdata[index + s];
  }
  __syncthreads();
}
```

이 패턴은 여전히 `blockDim.x`가 2의 거듭제곱이라는 가정 위에 서 있지만, 한 워프의 스레드들이 연속된 공유 메모리 뱅크를 접근하므로 Stage 07보다 다이버전스가 적고 `%` 연산이 사라집니다.

## 실습

### 목표

Stage 07과 동일한 공유 메모리 기반 reduction 커널에 **strided index interleaved 루프**를 도입한다.

### 단계

1. reduction 루프를 위 코드처럼 `index = 2 * s * tid` 계산 방식으로 교체한다. (`tid`는 `threadIdx.x` 캐시에 머물도록 지역 변수로 둔다.)
2. 각 반복마다 `__syncthreads()`를 호출해 `index`와 `index + s`의 공유 메모리 업데이트가 끝났는지 보장한다.

### 기대 출력

```
Stage 8 reduction matches reference ✅

Input size: 16777216 elements
CPU sum : <cpu_sum>
GPU sum : <gpu_sum>
Relative error: <error>

Timing:
  CPU time : <cpu_ms> ms
  GPU time : <gpu_ms> ms
```
