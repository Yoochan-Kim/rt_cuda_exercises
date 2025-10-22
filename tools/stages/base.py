from dataclasses import dataclass
from pathlib import Path
from typing import Callable


Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class StageInfo:
    """Holds verification metadata for a single stage."""

    stage_id: str
    exercise_src: Path
    answer_src: Path
    normalizer: Normalizer
    expected_output: str

