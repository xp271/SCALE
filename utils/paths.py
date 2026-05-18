"""Path helpers and well-known directory names used across packages."""
from __future__ import annotations

import sys
from pathlib import Path

# Default repo layout under the project root. LightCompress and LLM-sycophancy
# are expected to live under the quantization/ and evaluation/ packages
# respectively after the folder redesign; callers can override via config keys
# `llmc_dir` and `syco_repo_dir`.
DIR_LIGHTCOMPRESS = "quantization/LightCompress"
DIR_LLM_SYCOPHANCY = "evaluation/LLM-sycophancy"
DIR_SYCO_SCRIPT = "experiments/behavioral_analysis"


def resolve_path(value: str | None, config_dir: Path) -> Path | None:
    """Resolve a config path: absolute kept as-is, relative anchored at config_dir."""
    if not value or value.startswith("/"):
        return Path(value) if value else None
    return (config_dir / value).resolve()


def fs_safe_label(s: str) -> str:
    """HF org/model paths contain '/'; not valid as a single dir or filename segment."""
    return s.replace("\\", "_").replace("/", "_")


def resolve_path_roots(cfg: dict, config_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve (cache_root, save_root, result_root) from config with defaults.

    New config layout uses three explicit keys:
      - ``cache_root``  (default ``cache``):  HF cache + quantized model dirs.
      - ``save_root``   (default ``save``):   syco pkl outputs.
      - ``result_root`` (default ``result``): figure outputs.

    Legacy compat: if ``cache_root`` is absent but the old ``save_root`` key is
    set (single root semantic), we treat it as the cache_root and emit a
    deprecation note. The new save_root then defaults to ``save`` separately
    so the two no longer collide.
    """
    cache_raw = cfg.get("cache_root")
    save_raw = cfg.get("save_root")
    result_raw = cfg.get("result_root")

    if cache_raw is None and save_raw is not None and result_raw is None:
        print(
            "[config] legacy 'save_root' detected without 'cache_root'/'result_root'; "
            "treating it as cache_root for backward compatibility. "
            "Please migrate to explicit cache_root/save_root/result_root keys.",
            file=sys.stderr,
        )
        cache_root = resolve_path(save_raw, config_dir) or (config_dir / "cache").resolve()
        save_root = (config_dir / "save").resolve()
        result_root = (config_dir / "result").resolve()
    else:
        cache_root = resolve_path(cache_raw or "cache", config_dir) or (config_dir / "cache").resolve()
        save_root = resolve_path(save_raw or "save", config_dir) or (config_dir / "save").resolve()
        result_root = resolve_path(result_raw or "result", config_dir) or (config_dir / "result").resolve()
    return cache_root, save_root, result_root
