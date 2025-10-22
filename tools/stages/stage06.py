from pathlib import Path

from .base import StageInfo

ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = []
    for line in output.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        # Skip performance lines (CPU/GPU time, Speedup) as they vary
        if any(x in line for x in ["CPU Time", "GPU Time", "Speedup", "Performance Comparison"]):
            continue
        lines.append(line)
    return "\n".join(lines)


EXPECTED_OUTPUT = """Stage 6 matrix multiplication matches reference ✅"""

STAGE_INFO = StageInfo(
    stage_id="06",
    exercise_src=ROOT / "exercises" / "stage06" / "main.cu",
    answer_src=ROOT / "answers" / "stage06" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)
