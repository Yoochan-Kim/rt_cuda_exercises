#!/usr/bin/env python3

import argparse
import sys

from tools.verify_stage import verify_stages, verify_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA learning stage build and verification tool",
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Round number to execute. Round 1 runs stages 00-06, Round 2 runs stages 07-09.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Stage number to verify (e.g., 00). If not specified, verifies all stages.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available stage numbers.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Verify answer code instead of exercise code (builds and validates answers).",
    )
    parser.add_argument(
        "--skip_cpu",
        action="store_true",
        help="Build with SKIP_CPU flag and compare GPU output directly (no CPU reference validation).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full GPU output streams when running with --skip_cpu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    round_stages_map = {
        1: [f"{i:02d}" for i in range(7)],  # 00-06
        2: ["07", "08", "09"],
    }

    round_stages = round_stages_map.get(args.round)
    if round_stages is None:
        print(f"Round {args.round} is not configured yet.")
        return 1


    if args.list:
        print("Available stages:", " ".join(round_stages))
        return 0

    if args.stage:
        if args.stage not in round_stages:
            print(f"Stage {args.stage} is not part of round {args.round}.")
            return 1
        targets = [args.stage]
    else:
        targets = round_stages

    if args.answer:
        success = verify_answers(targets, skip_cpu=args.skip_cpu, verbose=args.verbose)
    else:
        success = verify_stages(targets, skip_cpu=args.skip_cpu, verbose=args.verbose)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
