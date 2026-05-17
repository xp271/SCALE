"""YAML config loading helpers."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config; relative paths are resolved against cwd."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
