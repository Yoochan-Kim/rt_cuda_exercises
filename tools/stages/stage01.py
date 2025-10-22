from pathlib import Path
from typing import List

from .base import StageInfo


ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    normalized: List[str] = []

    for line in lines:
        if line.startswith("Stage 1 results"):
            prefix, _, value_part = line.partition(":")
            numbers: List[int] = []
            parse_failed = False
            for token in value_part.split():
                try:
                    numbers.append(int(token))
                except ValueError:
                    parse_failed = True
                    break
            if parse_failed or not numbers:
                normalized.append(line)
            else:
                numbers.sort()
                normalized.append(f"{prefix}: {' '.join(str(n) for n in numbers)}")
        else:
            normalized.append(line)

    return "\n".join(normalized)


EXPECTED_OUTPUT = """Stage 1 thread ID write test passed ✅"""

STAGE_INFO = StageInfo(
    stage_id="01",
    exercise_src=ROOT / "exercises" / "stage01" / "main.cu",
    answer_src=ROOT / "answers" / "stage01" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)

