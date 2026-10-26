"""One instruction, repeated fresh RGB-D pick-and-place cycles.

The first safe pick still requires G/Space confirmation. Subsequent cycles
advance automatically only after a successful release and return to ready.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vlm_grasp_robot import main  # noqa: E402
from utils.vlm_sequence_utils import SequencePipeline  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(
        pipeline_factory=SequencePipeline,
        default_output_dir=PROJECT_ROOT / "outputs/vlm_sequence_robot",
        window_title="VLM Sequence Robot - REAL CONTROL",
        workflow_description=(
            "[连续任务] 首轮通过检查后按 G/Space 确认一次；以后每次放置并返回 ready，"
            "自动重新拍 RGB-D、清点盒外目标、计算抓取/放置位姿并执行。"
            "每轮按新图重新清点盒外目标；数量不减少也继续按原指令规划。"
            "连续两张新图确认盒外无目标后返回 ready 位并自动退出。"
            "安全检查失败或达到步数上限时暂停。"
            "抓取反馈未确认且已回 ready 时暂停，需按 g 重拍重算或按 r 放弃本轮。"
            "视觉失败可按 g 保留原指令重试，r 放弃，q 退出。"
        ),
        sequence_mode=True,
    ))
