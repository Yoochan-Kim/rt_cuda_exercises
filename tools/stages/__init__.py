from .base import StageInfo
from .stage00 import STAGE_INFO as stage00_info
from .stage01 import STAGE_INFO as stage01_info
from .stage02 import STAGE_INFO as stage02_info
from .stage03 import STAGE_INFO as stage03_info
from .stage04 import STAGE_INFO as stage04_info
from .stage05 import STAGE_INFO as stage05_info
from .stage06 import STAGE_INFO as stage06_info
from .stage07 import STAGE_INFO as stage07_info
from .stage08 import STAGE_INFO as stage08_info
from .stage09 import STAGE_INFO as stage09_info
from .stage10 import STAGE_INFO as stage10_info
from .stage11 import STAGE_INFO as stage11_info

STAGES = {
    stage00_info.stage_id: stage00_info,
    stage01_info.stage_id: stage01_info,
    stage02_info.stage_id: stage02_info,
    stage03_info.stage_id: stage03_info,
    stage04_info.stage_id: stage04_info,
    stage05_info.stage_id: stage05_info,
    stage06_info.stage_id: stage06_info,
    stage07_info.stage_id: stage07_info,
    stage08_info.stage_id: stage08_info,
    stage09_info.stage_id: stage09_info,
    stage10_info.stage_id: stage10_info,
    stage11_info.stage_id: stage11_info,
}

__all__ = ["StageInfo", "STAGES"]
