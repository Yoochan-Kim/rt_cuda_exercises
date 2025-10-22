from pathlib import Path

from .base import StageInfo

ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines)


EXPECTED_OUTPUT = """Stage 3 matrix add matches reference ✅"""

STAGE_INFO = StageInfo(
    stage_id="03",
    exercise_src=ROOT / "exercises" / "stage03" / "main.cu",
    answer_src=ROOT / "answers" / "stage03" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)

