"""GraspNet grasp followed by set.py's box localization and placement.

Usage:
    python scripts/grasp_set.py
    python scripts/grasp_set.py --target-class "yellow banana"
    python scripts/grasp_set.py --config config/default.yaml --dry-run

G/Space: pick and place. R: resume preview. Q/Esc: home and exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.grasp import main  # noqa: E402


if __name__ == "__main__":
    try:
        raise SystemExit(main(place_in_box=True))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
