from pathlib import Path
from typing import List

from .base import StageInfo


ROOT = Path(__file__).resolve().parents[2]


def _normalize(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    host_lines: List[str] = []
    gpu_lines: List[str] = []
    misc_lines: List[str] = []

    for line in lines:
        if line.startswith("Hello from the GPU"):
            gpu_lines.append(line)
        elif line.startswith("Hello from the CPU"):
            host_lines.append(line)
        else:
            misc_lines.append(line)

    gpu_lines.sort()
    normalized = host_lines + misc_lines + gpu_lines
    return "\n".join(normalized)


EXPECTED_OUTPUT = """Hello from the CPU (host function)!
Hello from the GPU (block 0, thread 0)!
Hello from the GPU (block 0, thread 1)!
Hello from the GPU (block 0, thread 2)!
Hello from the GPU (block 0, thread 3)!
Hello from the GPU (block 0, thread 4)!
Hello from the GPU (block 0, thread 5)!
Hello from the GPU (block 0, thread 6)!
Hello from the GPU (block 0, thread 7)!
Hello from the GPU (block 0, thread 8)!
Hello from the GPU (block 0, thread 9)!"""

STAGE_INFO = StageInfo(
    stage_id="00",
    exercise_src=ROOT / "exercises" / "stage00" / "main.cu",
    answer_src=ROOT / "answers" / "stage00" / "main.cu",
    normalizer=_normalize,
    expected_output=EXPECTED_OUTPUT,
)

