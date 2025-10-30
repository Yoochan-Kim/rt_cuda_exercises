#!/usr/bin/env python3

"""Generate golden reference outputs for SKIP_CPU mode using numpy.

This script generates expected outputs by computing results in Python,
without running any GPU code. Results are saved as pickled numpy arrays.
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "references"


def _mt19937_uint32(seed: int, size: int) -> np.ndarray:
    """Replicates std::mt19937 sequence producing 32-bit outputs."""
    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    state = [0] * N
    state[0] = seed & 0xFFFFFFFF
    for i in range(1, N):
        state[i] = (1812433253 * (state[i - 1] ^ (state[i - 1] >> 30)) + i) & 0xFFFFFFFF

    index = N
    result = np.empty(size, dtype=np.uint32)

    for j in range(size):
        if index >= N:
            for i in range(N):
                y = (state[i] & UPPER_MASK) + (state[(i + 1) % N] & LOWER_MASK)
                state[i] = state[(i + M) % N] ^ (y >> 1)
                if y & 0x1:
                    state[i] ^= MATRIX_A
            index = 0

        y = state[index]
        index += 1

        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)

        result[j] = y & 0xFFFFFFFF

    return result

def _mt19937_float32(seed: int, size: int) -> np.ndarray:
    """
    Replicates std::mt19937 sequence producing float outputs in [0.0, 1.0).
    Uses the output from the 32-bit integer generator.
    """
    uint_results = _mt19937_uint32(seed, size)
    float_results = uint_results.astype(np.float32) / 4294967296.0
    
    return float_results

def generate_stage02_reference() -> np.ndarray:
    """Stage 2: Vector addition."""
    k_blocks = 8
    k_threads = 256
    k_element_count = k_blocks * k_threads

    a = np.arange(k_element_count, dtype=np.float32)
    b = np.arange(k_element_count, 0, -1, dtype=np.float32)
    c = a + b

    return c


def generate_stage03_reference() -> np.ndarray:
    """Stage 3: Matrix addition (8x8)."""
    k_matrix_size = 8

    indices = np.arange(k_matrix_size * k_matrix_size, dtype=np.float32)
    a = 1.0 * indices
    b = -0.5 * indices
    c = a + b

    return c


def generate_stage04_reference() -> np.ndarray:
    """Stage 4: Large matrix addition (500x500)."""
    k_matrix_size = 500

    indices = np.arange(k_matrix_size * k_matrix_size, dtype=np.float32)
    a = 1.0 * indices
    b = -0.5 * indices
    c = a + b

    return c


def generate_stage05_reference() -> np.ndarray:
    """Stage 5: Matrix multiplication without shared memory (M=5120, K=2048, N=3072)."""
    M = 5120
    K = 2048
    N = 3072

    # Initialize matrices A (M x K) and B (K x N)
    # Fill pattern: for each (row, col), idx = row * width + col, value = scale * (idx % 100)

    # Create index matrices using broadcasting
    rows_a = np.arange(M, dtype=np.int32).reshape(M, 1)
    cols_a = np.arange(K, dtype=np.int32).reshape(1, K)
    indices_a = rows_a * K + cols_a
    A = 0.01 * (indices_a % 100).astype(np.float32)

    rows_b = np.arange(K, dtype=np.int32).reshape(K, 1)
    cols_b = np.arange(N, dtype=np.int32).reshape(1, N)
    indices_b = rows_b * N + cols_b
    B = 0.02 * (indices_b % 100).astype(np.float32)

    # Compute C = A @ B (M x N)
    C = A @ B

    return C.flatten()


def generate_stage06_reference() -> np.ndarray:
    """Stage 6: Matrix multiplication with shared memory (M=5120, K=2048, N=3072)."""
    # Same computation as stage 5, just different GPU implementation
    return generate_stage05_reference()


def generate_stage07_reference() -> np.ndarray:
    """Stage 7: Warp divergence benchmark (divergent, split)."""
    k_element_count = 1 << 24
    k_heavy_iterations = 96
    k_light_iterations = 8

    seed = 12345
    raw = _mt19937_uint32(seed, k_element_count)
    inv_max = np.float32(1.0) / np.float32(np.iinfo(np.uint32).max)
    base_values = raw.astype(np.float32) * inv_max * np.float32(2.0) - np.float32(1.0)

    indices = np.arange(k_element_count, dtype=np.int32)
    even_indices = np.arange(0, k_element_count, 2, dtype=np.int32)
    odd_indices = np.arange(1, k_element_count, 2, dtype=np.int32)
    even_mask = (indices & 1) == 0

    def apply_heavy(values: np.ndarray, iterations: int) -> np.ndarray:
        out = values.astype(np.float32, copy=True)
        for _ in range(iterations):
            out = np.sin(out) * np.cos(out) + np.float32(0.5)
            out = out.astype(np.float32, copy=False)
        return out

    def apply_light(values: np.ndarray, iterations: int) -> np.ndarray:
        out = values.astype(np.float32, copy=True)
        for _ in range(iterations):
            out = out * np.float32(1.125) + np.float32(0.25)
            out = out.astype(np.float32, copy=False)
        return out

    # Divergent branch: even indices heavy work, odd indices light work.
    divergent = base_values.copy()
    divergent_even = apply_heavy(divergent[even_indices], k_heavy_iterations)
    divergent_odd = apply_light(divergent[odd_indices], k_light_iterations)
    divergent[even_indices] = divergent_even
    divergent[odd_indices] = divergent_odd
    divergent_sum = np.float32(np.sum(divergent, dtype=np.float64))

    # Split kernels: process even and odd ranges separately.
    split = base_values.copy()
    split_even = apply_heavy(split[even_indices], k_heavy_iterations)
    split_odd = apply_light(split[odd_indices], k_light_iterations)
    split[even_indices] = split_even
    split[odd_indices] = split_odd
    split_sum = np.float32(np.sum(split, dtype=np.float64))

    return np.array([divergent_sum, split_sum], dtype=np.float32)


def generate_stage08_reference() -> np.ndarray:
    """Stage 8: Early-exit divergence (divergent, compaction)."""
    k_element_count = 1 << 24
    k_energy_threshold = 0.7
    k_compute_iterations = 1000

    # Generate particles matching main.cu logic
    seed = 12345
    raw = _mt19937_float32(seed, 4*k_element_count)
    energy = raw[:k_element_count]

    u_x = raw[k_element_count:2*k_element_count]
    x = -10.0 + 20.0 * u_x
    x = np.array(x, dtype=np.float32)

    u_y = raw[2*k_element_count:3*k_element_count]
    y = -10.0 + 20.0 * u_y
    y = np.array(y, dtype=np.float32)

    u_z = raw[3*k_element_count:]
    z = -10.0 + 20.0 * u_z
    z = np.array(z, dtype=np.float32)

    # Vectorized expensive computation (much faster than loop)
    # Only compute for active particles
    active_mask = energy >= k_energy_threshold
    active_indices = np.where(active_mask)[0]

    divergent_output = np.zeros(k_element_count, dtype=np.float32)

    if len(active_indices) > 0:
        e = energy[active_indices]
        x_active = x[active_indices]
        y_active = y[active_indices]
        z_active = z[active_indices]

        result = np.zeros_like(e)
        for i in range(k_compute_iterations):
            temp = e + x_active * np.sin(y_active + i * np.float32(0.01))
            result += temp * np.cos(z_active + i * np.float32(0.01))
            result = result * np.float32(0.99) + temp * np.float32(0.01)

        divergent_output[active_indices] = result

    divergent_sum = float(np.sum(divergent_output, dtype=np.float64))

    # Compaction produces same result, just different implementation
    compaction_sum = divergent_sum

    return np.array([divergent_sum, compaction_sum], dtype=np.float64)


STAGE_GENERATORS = {
    "02": generate_stage02_reference,
    "03": generate_stage03_reference,
    "04": generate_stage04_reference,
    "05": generate_stage05_reference,
    "06": generate_stage06_reference,
    "07": generate_stage07_reference,
    "08": generate_stage08_reference,
}


def generate_reference_for_stage(stage_id: str) -> bool:
    """Generate golden reference for a specific stage."""
    if stage_id not in STAGE_GENERATORS:
        print(f"[Stage {stage_id}] No generator available (stages 00, 01 don't need skip_cpu mode)")
        return True

    print(f"[Stage {stage_id}] Generating golden reference...")

    try:
        result = STAGE_GENERATORS[stage_id]()

        # Convert numpy array to plain Python list for pickle (no numpy dependency on read)
        result_list = result.tolist()

        # Save output to reference file as pickle
        REFERENCE_DIR.mkdir(exist_ok=True)
        reference_file = REFERENCE_DIR / f"stage{stage_id}_skip_cpu.pkl"

        with open(reference_file, "wb") as f:
            pickle.dump(result_list, f)

        size_kb = reference_file.stat().st_size / 1024
        print(f"[Stage {stage_id}] ✅ Reference saved ({len(result_list)} elements, {size_kb:.1f} KB) -> {reference_file.relative_to(ROOT)}")
        return True

    except Exception as e:
        print(f"[Stage {stage_id}] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate golden reference outputs for SKIP_CPU mode using pure Python",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Stage number to generate reference for (e.g., 02). If not specified, generates for all stages.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.stage:
        stage_ids = [args.stage]
    else:
        stage_ids = sorted(STAGE_GENERATORS.keys())

    all_passed = True
    for stage_id in stage_ids:
        passed = generate_reference_for_stage(stage_id)
        all_passed = all_passed and passed

    if all_passed:
        print("\n✅ All golden references generated successfully")
    else:
        print("\n❌ Some references failed to generate")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
