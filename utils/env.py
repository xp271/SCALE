"""Environment helpers: CUDA visibility, HF cache redirection, debug flag parsing."""
from __future__ import annotations

import os
from pathlib import Path


def truthy_inference_debug(v) -> bool:
    """Matches run_syco --debug_inference / RUN_SYCO_DEBUG: enable for true, 1, 2, yes, on."""
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, int):
        return v != 0
    s = str(v).strip().lower()
    return s not in ("", "0", "false", "no", "off")


def cuda_visible_devices_from_config(val: str | int | None) -> str | None:
    """Return CUDA_VISIBLE_DEVICES string, or None to leave the env unset (all GPUs visible).

    pipeline cuda_device / --gpu may use auto|all|none to mean: do not narrow visibility.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in ("auto", "all", "none", "default"):
        return None
    return s


def setup_hf_cache_env(cache_root: Path | None) -> None:
    """Redirect HF cache / TMPDIR under cache_root to avoid filling the home disk."""
    if cache_root is None:
        return
    hf_cache_dir = cache_root / "_hf_cache"
    tmp_dir = cache_root / "_tmp"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))
