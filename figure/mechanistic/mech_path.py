"""将 mechanistic_analysis 置于 ``sys.path`` 首部，便于 ``import kl_divergence_compute`` 等。"""
from __future__ import annotations

import sys
from pathlib import Path

_MECH_RELATIVE = Path("evaluation/LLM-sycophancy/experiments/mechanistic_analysis")


def resolve_mechanistic_analysis_dir() -> Path:
    from figure.mechanistic.repo_root import resolve_project_root

    root = resolve_project_root(Path(__file__))
    return (root / _MECH_RELATIVE).resolve()


def prepend_mech_syspath(mech_dir: Path | None = None) -> Path:
    d = resolve_mechanistic_analysis_dir() if mech_dir is None else Path(mech_dir).resolve()
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    return d
