from pathlib import Path

from .base import StageInfo

ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Input size:"):
            continue
        if line.startswith("CPU sum"):
            continue
        if line.startswith("GPU sum"):
            continue
        if line.startswith("Relative error"):
            continue
        if line.startswith("Timing"):
            continue
        if line.startswith("CPU time"):
            continue
        if line.startswith("GPU time"):
            continue
        lines.append(line)
    return "\n".join(lines)


EXPECTED_OUTPUT = """Stage 7 reduction matches reference ✅"""

STAGE_INFO = StageInfo(
    stage_id="07",
    exercise_src=ROOT / "exercises" / "stage07" / "main.cu",
    answer_src=ROOT / "answers" / "stage07" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)
