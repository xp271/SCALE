"""Resolve eva_syc project root and ensure it is importable."""
from __future__ import annotations

import sys
from pathlib import Path


def resolve_project_root(anchor_file: Path) -> Path:
    """Walk parents until we find a directory that looks like the repo root."""
    start = anchor_file.resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "figure").is_dir() and (parent / "evaluation").is_dir():
            return parent
    raise RuntimeError(f"Cannot find project root upward from {start}")


def ensure_project_syspath(origin: Path) -> Path:
    """Prepend repo root so ``import figure.mechanistic`` works."""
    root = resolve_project_root(origin)
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
