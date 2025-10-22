#!/usr/bin/env python3

"""Utility helpers to verify CUDA learning stages."""

import difflib
import pickle
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .stages import STAGES, StageInfo


ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
REFERENCE_DIR = ROOT / "references"


def _run_command(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _build_with_make(stage: StageInfo, skip_cpu: bool = False) -> bool:
    result = _run_command(["make", f"stage{stage.stage_id}", f"SKIP_CPU={1 if skip_cpu else 0}"])
    if result.returncode != 0:
        print(f"[Stage {stage.stage_id}] Build failed (make stage{stage.stage_id})")
        if result.stdout:
            print("stdout:")
            print(result.stdout.rstrip())
        if result.stderr:
            print("stderr:")
            print(result.stderr.rstrip())
        return False
    return True


def _run_binary(binary: Path) -> subprocess.CompletedProcess:
    return _run_command([str(binary)])


def _build_target(stage: StageInfo, target: str, skip_cpu: bool = False) -> bool:
    # Remove existing binary to force rebuild (SKIP_CPU flag changes require rebuild)
    binary = BIN_DIR / f"stage{stage.stage_id}_{target}"
    if binary.exists():
        binary.unlink()

    cmd = ["make", f"stage{stage.stage_id}_{target}"]
    if skip_cpu:
        cmd.append("SKIP_CPU=1")
    result = _run_command(cmd)
    if result.returncode != 0:
        print(f"[Stage {stage.stage_id}] Build failed (make stage{stage.stage_id}_{target})")
        if result.stdout:
            print("stdout:")
            print(result.stdout.rstrip())
        if result.stderr:
            print("stderr:")
            print(result.stderr.rstrip())
        return False
    return True


def _verify_target(stage: StageInfo, target: str, skip_cpu: bool = False, verbose: bool = False) -> bool:
    src = stage.exercise_src if target == "exercise" else stage.answer_src
    label = target.capitalize()

    if not src.exists():
        print(f"[Stage {stage.stage_id}] {label} source not found: {src}")
        return False

    # In skip_cpu mode, check reference file exists BEFORE building
    # (Skip stages 00 and 01 as they don't have CPU validation anyway)
    skip_cpu_actual = skip_cpu and stage.stage_id not in ["00", "01"]

    if skip_cpu_actual:
        reference_file = REFERENCE_DIR / f"stage{stage.stage_id}_skip_cpu.pkl"
        if not reference_file.exists():
            print(f"[Stage {stage.stage_id}] ❌ Reference file not found: {reference_file}")
            print(f"Run 'python generate_reference.py --stage {stage.stage_id}' to generate it first")
            return False

    if not _build_target(stage, target, skip_cpu_actual):
        return False

    binary = BIN_DIR / f"stage{stage.stage_id}_{target}"
    run_result = _run_binary(binary)

    if run_result.returncode != 0:
        print(f"[Stage {stage.stage_id}] {label} execution failed (code {run_result.returncode})")
        if run_result.stdout:
            print(run_result.stdout.rstrip())
        if run_result.stderr:
            print(run_result.stderr.rstrip())
        return False

    if skip_cpu_actual:
        # In skip_cpu mode, load reference from pickle file (already checked existence above)
        reference_file = REFERENCE_DIR / f"stage{stage.stage_id}_skip_cpu.pkl"

        with open(reference_file, "rb") as f:
            reference_list = pickle.load(f)

        # Parse GPU output into list
        try:
            gpu_values = [float(line.strip()) for line in run_result.stdout.strip().splitlines() if line.strip()]
        except Exception as e:
            print(f"[Stage {stage.stage_id}] ❌ Failed to parse GPU output: {e}")
            return False

        # Compare with tolerance
        if len(gpu_values) != len(reference_list):
            print(f"[Stage {stage.stage_id}] ❌ Size mismatch: GPU={len(gpu_values)}, Reference={len(reference_list)}")
            return False

        # Compare with tolerance
        tolerance = 1e-3
        mismatches = []
        for i, (gpu_val, ref_val) in enumerate(zip(gpu_values, reference_list)):
            diff = abs(gpu_val - ref_val)
            rel_err = diff / max(abs(ref_val), 1e-6)
            if diff > tolerance and rel_err > tolerance:
                mismatches.append((i, gpu_val, ref_val, diff))

        if len(mismatches) == 0:
            if verbose:
                print(f"[Stage {stage.stage_id}] ✅ {label} passed (GPU output matches reference)")
                print(run_result.stdout.rstrip())
            else:
                print(f"[Stage {stage.stage_id}] ✅ {label} passed (GPU output matches reference)")
            return True

        # Show mismatches
        print(f"[Stage {stage.stage_id}] ❌ {label} GPU output mismatch")
        print(f"  Mismatches: {len(mismatches)}/{len(gpu_values)} ({100.0*len(mismatches)/len(gpu_values):.2f}%)")

        # Show first few mismatches
        print("  First mismatches:")
        for i, gpu_val, ref_val, diff in mismatches[:10]:
            print(f"    [{i}] GPU={gpu_val:.6f}, Ref={ref_val:.6f}, Diff={diff:.6e}")

        if verbose:
            print("\n---- Full GPU output ----")
            print(run_result.stdout.rstrip())

        return False
    else:
        # Normal mode: compare with expected output
        norm_output = stage.normalizer(run_result.stdout)
        norm_expected = stage.normalizer(stage.expected_output)

        if norm_output == norm_expected:
            if verbose:
                print(f"[Stage {stage.stage_id}] ✅ {label} passed")
                print(run_result.stdout.rstrip())
            else:
                print(f"[Stage {stage.stage_id}] ✅ {label} passed")
            return True

        print(f"[Stage {stage.stage_id}] ❌ {label} output mismatch")
        print(f"---- {label} (normalized) ----")
        print(norm_output)
        print("---- Expected (normalized) ----")
        print(norm_expected)

        diff = difflib.unified_diff(
            norm_expected.splitlines(),
            norm_output.splitlines(),
            fromfile="expected",
            tofile=target,
            lineterm="",
        )
        print("---- diff ----")
        for line in diff:
            print(line)

        if verbose:
            print("\n---- Full output ----")
            print(run_result.stdout.rstrip())

        return False


def _verify_stage(stage: StageInfo, skip_cpu: bool = False, verbose: bool = False) -> bool:
    return _verify_target(stage, "exercise", skip_cpu, verbose)


def verify_stage(stage_id: str, skip_cpu: bool = False, verbose: bool = False) -> bool:
    stage = STAGES.get(stage_id)
    if stage is None:
        print(f"[Stage {stage_id}] Configuration not found.")
        return False
    return _verify_stage(stage, skip_cpu, verbose)


def verify_stages(stage_ids: Optional[Iterable[str]] = None, skip_cpu: bool = False, verbose: bool = False) -> bool:
    if stage_ids is None:
        target_ids = sorted(STAGES.keys())
    else:
        target_ids = list(stage_ids)

    all_passed = True
    for stage_id in target_ids:
        passed = verify_stage(stage_id, skip_cpu, verbose)
        all_passed = all_passed and passed
    return all_passed


def _verify_answer(stage: StageInfo, skip_cpu: bool = False, verbose: bool = False) -> bool:
    return _verify_target(stage, "answer", skip_cpu, verbose)


def verify_answer(stage_id: str, skip_cpu: bool = False, verbose: bool = False) -> bool:
    stage = STAGES.get(stage_id)
    if stage is None:
        print(f"[Stage {stage_id}] Configuration not found.")
        return False
    return _verify_answer(stage, skip_cpu, verbose)


def verify_answers(stage_ids: Optional[Iterable[str]] = None, skip_cpu: bool = False, verbose: bool = False) -> bool:
    if stage_ids is None:
        target_ids = sorted(STAGES.keys())
    else:
        target_ids = list(stage_ids)

    all_passed = True
    for stage_id in target_ids:
        passed = verify_answer(stage_id, skip_cpu, verbose)
        all_passed = all_passed and passed
    return all_passed


def known_stages() -> List[str]:
    return sorted(STAGES.keys())

