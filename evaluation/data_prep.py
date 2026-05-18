"""Ensure LLM-sycophancy lib pkl files exist for a given data seed.

Wraps the two upstream helper scripts (``download_*.py`` and
``build_lib_from_raw.py``) so a fresh seed can be evaluated end-to-end without
manual data preparation. Also verifies that the Academic three-level inputs
exist when ``eval_authority_advanced`` is enabled.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_syco_data_for_seed(syco_repo: Path, seed: int, syco_cfg: dict) -> bool:
    """Download raw and run build_lib_from_raw if plain / opinion_only pkls for this seed are missing."""
    dataset = syco_cfg.get("dataset", "mmlu")
    data_slug = syco_cfg.get("data_slug") or dataset
    raw_rel = syco_cfg.get("raw_file", "raw_data/mmlu_raw.pkl")
    download_name = syco_cfg.get("download_script", "download_mmlu.py")
    extra = syco_cfg.get("build_lib_extra_args") or []
    if not isinstance(extra, list):
        extra = list(extra) if extra else []

    plain_pkl = syco_repo / "lib" / "plain" / f"{data_slug}_plain_{seed}.pkl"
    opinion_pkl = syco_repo / "lib" / "opinion_only" / "prefix" / f"{data_slug}_opinion_only_{seed}.pkl"
    if plain_pkl.exists() and opinion_pkl.exists():
        return True

    raw_pkl = (syco_repo / raw_rel).resolve() if not os.path.isabs(raw_rel) else Path(raw_rel)
    dg_dir = syco_repo / "experiments" / "data_generation"
    download_script = dg_dir / download_name
    if not raw_pkl.exists():
        if not download_script.exists():
            print(f"missing raw data and {download_script} not found", file=sys.stderr)
            return False
        raw_pkl.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading raw data to {raw_pkl} ({download_name})...")
        try:
            r = subprocess.run(
                [sys.executable, str(download_script), "--output", str(raw_pkl)],
                cwd=str(dg_dir),
                timeout=600,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"{download_name} failed: {r.stderr or r.stdout}", file=sys.stderr)
                return False
        except subprocess.TimeoutExpired:
            print(f"{download_name} timed out", file=sys.stderr)
            return False
        if not raw_pkl.exists():
            print(f"still missing after download: {raw_pkl}", file=sys.stderr)
            return False

    build_script = dg_dir / "build_lib_from_raw.py"
    if not build_script.exists():
        print(f"not found: {build_script}", file=sys.stderr)
        return False
    print(f"Building lib plain/opinion_only for seed={seed} (data_slug={data_slug})...")
    try:
        cmd = [
            sys.executable,
            str(build_script),
            "--seed",
            str(seed),
            "--raw_file",
            raw_rel,
        ]
        cmd.extend(str(x) for x in extra)
        r = subprocess.run(
            cmd,
            cwd=str(syco_repo),
            timeout=600,
            capture_output=True,
            text=True,
        )
        if not plain_pkl.exists() or not opinion_pkl.exists():
            if r.returncode != 0 and r.stderr:
                print(r.stderr, file=sys.stderr)
            print(
                f"pkls still missing after build: plain={plain_pkl.exists()}, opinion={opinion_pkl.exists()}",
                file=sys.stderr,
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        print("build_lib_from_raw timed out", file=sys.stderr)
        return False


def verify_academic_three_levels(syco_repo: Path, data_slug: str, seed: int) -> bool:
    """Check the three Academic-prefix pkls exist for ``eval_authority_advanced``.

    Returns True if all three (beginner / intermediate / advanced) are present;
    on failure prints a helpful message listing what was found and missing.
    """
    fp_dir = syco_repo / "lib" / "pov" / "prefix" / "first_pov"
    required_levels = ("beginner", "intermediate", "advanced")
    missing = []
    for level in required_levels:
        p = fp_dir / f"{data_slug}_academic_opinion_{level}_{seed}.pkl"
        if not p.exists():
            missing.append((level, p))
    if not missing:
        return True
    hint = ""
    if fp_dir.is_dir():
        all_glob = sorted(fp_dir.glob(f"{data_slug}_academic_opinion_*_{seed}.pkl"))
        if all_glob:
            names = ", ".join(p.name for p in all_glob[:12])
            more = " …" if len(all_glob) > 12 else ""
            hint = (
                f"\nFound in same dir ({data_slug}, seed={seed}): {names}{more}"
                f"\nRequired three levels: beginner / intermediate / advanced."
            )
    miss_text = "\n".join([f"- {lvl}: {path}" for lvl, path in missing])
    print(
        "eval_authority_advanced is on but Academic three-level input files are missing:\n"
        f"{miss_text}\n"
        f"Run generate_prefixes.py and build_lib_from_raw.py for seed={seed} (match build_lib_extra_args)."
        f"{hint}",
        file=sys.stderr,
    )
    return False
