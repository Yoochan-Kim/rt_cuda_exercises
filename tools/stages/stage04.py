from pathlib import Path

from .base import StageInfo

ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines)


EXPECTED_OUTPUT = """Stage 4 matrix add matches reference ✅"""

STAGE_INFO = StageInfo(
    stage_id="04",
    exercise_src=ROOT / "exercises" / "stage04" / "main.cu",
    answer_src=ROOT / "answers" / "stage04" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)
