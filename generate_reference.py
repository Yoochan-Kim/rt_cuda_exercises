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


STAGE_GENERATORS = {
    "02": generate_stage02_reference,
    "03": generate_stage03_reference,
    "04": generate_stage04_reference,
    "05": generate_stage05_reference,
    "06": generate_stage06_reference,
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
