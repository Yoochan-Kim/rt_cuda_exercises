# Stage 08: Early-Exit Warp Divergence

## 튜토리얼 목표

- 조기 종료(early-exit)로 인한 warp divergence 문제를 이해합니다.
- Stream compaction을 통해 divergence를 완전히 제거하는 최적화 기법을 학습합니다.

## Early-Exit Divergence란?

GPU에서 일부 스레드만 조건을 만족하여 조기 종료하는 경우, 나머지 스레드가 계속 작업하는 동안 종료된 스레드는 idle 상태로 대기합니다. 이것이 early-exit divergence입니다.

```cpp
if (particle.energy < threshold) {
    return;  // 일부 스레드만 여기서 종료
}
// 나머지 스레드만 비싼 계산 수행
expensiveComputation();
```

같은 warp 내에서 30%만 작업이 필요한 경우, GPU는 100% 시간을 사용하지만 30%의 효율만 얻게 됩니다.

## 두 가지 접근 방식

1. **Divergent Kernel (기존 코드)**
   - 모든 파티클을 처리하되, threshold 이하는 조기 종료합니다.
   - Active 30%인 경우: warp가 70% 시간을 낭비하며 대기합니다.

2. **Stream Compaction (최적화)**
   - Phase 1: Active 파티클만 dense array로 압축합니다.
   - Phase 2: 압축된 배열만 처리하므로 모든 스레드가 100% 활용됩니다.
   - 두 단계로 나뉘지만, 실제 계산은 필요한 만큼만 수행하므로 전체적으로 ~3배 빠릅니다.

## Stream Compaction의 작동 원리

**Phase 1: Filter and Compact**
```cpp
if (particle.energy >= threshold) {
    int pos = atomicAdd(outputCount, 1);
    compacted[pos] = particle;
    indices[pos] = originalIndex;
}
```
- 1M 파티클 → 300K active만 압축
- 가벼운 연산이므로 divergence 영향이 적음

**Phase 2: Process Compacted**
```cpp
// 모든 스레드가 valid work를 가짐
particle = compacted[idx];
result = expensiveComputation(particle);
output[indices[idx]] = result;
```
- 300K 스레드만 실행
- Divergence 없음: 모든 스레드가 동일한 작업 수행
- 결과는 원래 위치에 기록

## 실습 가이드

1. `todo.cu`에서 두 개의 stream compaction 커널을 완성합니다:
   - `compactKernel`: active 파티클만 필터링하고 atomicAdd로 압축
   - `processCompactedKernel`: 압축된 배열 처리하고 원래 위치에 결과 저장

2. 각 커널에서 `idx >= count` 조건을 반드시 확인합니다.

3. Phase 1에서는 원본 인덱스를 `indices` 배열에 저장해야 Phase 2에서 올바른 위치에 결과를 쓸 수 있습니다.

## 기대 출력

```
Stage 8 early-exit divergence ✅

Dataset:
  Particles: 16777216
  Block size: 512
  Energy threshold: 0.7
  Compute iterations: 1000
  Active particles: 5033164 (30.1%)

Results (avg over 100 runs):
  Divergent (early-exit) : <divergent-ms> ms | checksum <divergent-sum>
  Stream compaction      : <compaction-ms> ms | checksum <compaction-sum>

Speedup:
  Stream compaction : <speedup>x faster
```