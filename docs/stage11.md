# Stage 11: Unroll the Last Warp

## 튜토리얼 목표

- Stage 10 이후 남은 병목이 전역 메모리보다 **명령어 수(Instruction Overhead)** 에 있다는 사실을 짚어 본다.
- 마지막 warp가 수행하는 반복을 수동으로 언롤해 조건문과 동기화 호출을 걷어낸다.
- warp-synchronous 구간에서 공유 메모리를 다룰 때 `volatile` 포인터가 필요한 이유를 이해한다.

## 남은 병목: Instruction Overhead

Stage 10은 `first add during load`로 글로벌 로드 대비 덧셈을 늘려 병목을 메모리에서 떼어냈습니다. 이제 한 블록이 데이터를 모두 공유 메모리로 옮긴 뒤에는 반복마다 활성 스레드 수가 절반씩 줄어들고, 결국 마지막 반복에서는 하나의 warp만 남습니다. 그럼에도 불구하고 모든 반복에서 `if (tid < stride)` 조건을 평가하고 `__syncthreads()`로 전체 블록을 대기시키기 때문에, **마지막 몇 단계는 연산 자체보다 부수적인 명령어가 더 많은 상황**이 됩니다.

Warp 내부는 SIMT 방식으로 같은 명령어를 동시에 실행하므로, 동일한 warp만 남은 상태에서는 추가 동기화나 분기 없이 공유 메모리 값을 안전하게 읽을 수 있습니다. 따라서 마지막 warp가 맡는 6개 반복(`stride = 32, 16, 8, 4, 2, 1`)을 직접 언롤하면, 반복 제어와 조건문, 동기화 호출을 모두 제거할 수 있습니다.

## 마지막 warp 언롤링 전략

1. **메인 루프는 32 초과 구간만 담당**합니다. `for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)` 형태로 수정해, 최소 두 warp가 활동하는 구간에서만 조건문과 `__syncthreads()`를 유지합니다.
2. **warp-synchronous 구간을 별도 함수(또는 코드 블록)** 로 분리합니다. `if (tid < 32)` 조건을 사용해 마지막 warp만 진입시키고, 다음과 같이 직접 덧셈을 나열합니다.

   ```cpp
   volatile float* vs = sdata;
   vs[tid] += vs[tid + 32];
   vs[tid] += vs[tid + 16];
   vs[tid] += vs[tid + 8];
   vs[tid] += vs[tid + 4];
   vs[tid] += vs[tid + 2];
   vs[tid] += vs[tid + 1];
   ```

   `volatile` 키워드를 사용하지 않으면 컴파일러가 메모리 접근을 캐시하거나 재주문(reorder)해 Warp 내에서 데이터 종속성이 깨질 수 있습니다. 동기화 없이 공유 메모리를 재차 읽기 때문에, 이 키워드는 Stage 11에서 correctness를 지켜 주는 필수 요소입니다.
3. **시작부와 마무리는 Stage 10과 동일**합니다. 두 값을 읽어 더한 뒤 공유 메모리에 적재하고, 마지막에 `tid == 0` 스레드가 `sdata[0]`을 전역 메모리에 저장합니다. 경계 처리를 포함한 로드 로직도 그대로 유지해야 합니다.

## 실습

### 목표

Stage 10 커널을 확장해 마지막 warp의 반복을 언롤하고, 조건문·동기화 오버헤드를 제거한다.

### 단계

1. 공유 메모리 적재와 경계 검사는 Stage 10 구현을 재사용한다.
2. Sequential addressing 루프의 종료 조건을 `stride > 32`로 바꾸고, 반복 끝마다 `__syncthreads()`를 호출한다.
3. `if (tid < 32)` 블록 안에서 `volatile` 포인터를 사용해 마지막 6단계 덧셈을 수동으로 언롤한다. 이 구간에서는 추가 동기화를 호출하지 않는다.
4. 루프와 언롤링이 끝나면 `tid == 0` 스레드가 블록 합을 `g_odata[blockIdx.x]`에 기록한다.

### 기대 출력

```
Stage 11 reduction matches reference ✅

Input size: 16777216 elements
CPU sum : <cpu_sum>
GPU sum : <gpu_sum>
Relative error: <error>

Timing:
  CPU time : <cpu_ms> ms
  GPU time : <gpu_ms> ms
```
