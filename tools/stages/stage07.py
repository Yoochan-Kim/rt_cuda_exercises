from pathlib import Path

from .base import StageInfo

ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


EXPECTED_OUTPUT = "Stage 7 warp divergence benchmark ✅"

STAGE_INFO = StageInfo(
    stage_id="07",
    exercise_src=ROOT / "exercises" / "stage07" / "main.cu",
    answer_src=ROOT / "answers" / "stage07" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)
