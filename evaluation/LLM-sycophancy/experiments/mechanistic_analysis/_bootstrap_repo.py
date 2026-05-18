"""Prepend repo root onto ``sys.path`` so ``import figure...`` works from mech scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_syspath(*, origin_file: Path) -> Path:
    start = Path(origin_file).resolve()
    for p in [start.parent, *start.parents]:
        if (p / "figure").is_dir() and (p / "evaluation").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return p
    raise RuntimeError(
        "无法在父目录中找到项目根目录（需要同时包含 figure/ 与 evaluation/）。"
        f"自 {start} 向上遍历失败。"
    )
