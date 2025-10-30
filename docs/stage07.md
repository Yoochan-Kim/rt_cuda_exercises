# Stage 07: Warp Divergence 기본 실험

## 튜토리얼 목표

- Warp 단위 실행(SIMT)과 분기(divergence)가 어떻게 성능에 영향을 주는지 이해합니다.
- 짝수/홀수 스레드가 다른 일을 할 때 발생하는 성능 손실을 직접 측정합니다.
- workload를 분리(split)하여 warp가 동일한 실행 경로를 갖도록 만드는 전략을 연습합니다.

## Warp Divergence란?

GPU는 동일한 warp에 속한 32개 스레드를 한 번에 스케줄링합니다. warp 내부 스레드가 서로 다른 분기 경로를 선택하면 하드웨어는 경로마다 순차적으로 명령어를 실행하고, 나머지 스레드는 마스크된 상태로 대기합니다. 즉, divergence가 발생하면 이론적으로 최대 32배까지 실행 시간이 늘어날 수 있습니다.

Stage 07에서는 `threadIdx`의 짝·홀 여부에 따라 연산량이 달라지는 커널을 확인합니다. 같은 warp 안에서 절반의 스레드는 무거운 연산(96회 루프)을, 나머지 스레드는 가벼운 연산(8회 루프)을 수행하도록 설정해 분기 비용을 직접 측정합니다.

## 두 가지 접근 방식

1. **Divergent Branch (기존 코드)**  
   - 짝수 스레드는 무거운 루프를, 홀수 스레드는 가벼운 루프를 실행합니다.  
   - 같은 warp 안에 서로 다른 분기 경로가 공존하므로 warp divergence가 가장 심합니다.

2. **Split Kernels (작업 분리)**  
   - 짝수 인덱스와 홀수 인덱스를 별도의 커널로 처리합니다.  
   - 각 커널은 균일한 작업만 포함하므로 warp divergence가 사라집니다. 커널을 두 번 실행하지만 warp 효율이 높아져 전체 시간이 단축됩니다.

두 전략을 CUDA 이벤트 기반 타이머로 여러 번 측정해 평균 실행 시간을 비교하고, 각 커널이 생성한 결과 벡터의 checksum도 함께 출력합니다.

## 실습 가이드

1. `todo.cu`에서 divergent/split 커널을 완성합니다.  
   - 짝수 인덱스는 `kHeavyIterations`번 `sinf/cosf` 연산을 반복합니다.  
   - 홀수 인덱스는 `kLightIterations`번 선형 변환을 수행합니다.
2. `main.cu`에서 각 커널을 여러 번 실행하고 CUDA 이벤트로 평균 시간을 계산합니다.
3. 디바이스 결과를 호스트로 복사한 뒤 double 누적으로 checksum을 계산합니다.
4. 성능(시간)과 정확도(checksum)가 모두 출력되도록 마무리합니다.

## 기대 출력

```
Stage 7 warp divergence benchmark ✅

Configuration:
  Elements: 1048576
  Block size: 256
  Heavy iterations: 96
  Light iterations: 8

Results (avg over 40 runs):
  Divergent branch : <divergent-ms> ms | checksum <divergent-sum>
  Split kernels    : <split-ms> ms | checksum <split-sum>

Speedup vs divergent:
  Split: <speedup>x
```
