"""Mechanistic plots: compute_decision_score.py and plot_kl_divergence.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PLOT_TIMEOUT = 300


def run_decision_score(
    *,
    mech_dir: Path,
    plain_paths: list[str],
    opinion_paths: list[str],
    out_plot: str,
) -> int:
    """Run compute_decision_score.py merging plain+opinion pkl lists."""
    cmd = [
        sys.executable, "compute_decision_score.py",
        "--plain", *plain_paths,
        "--opinion", *opinion_paths,
        "--out_plot", out_plot,
    ]
    print(f"[Plot] Running compute_decision_score.py -> {out_plot} ...")
    ret = subprocess.run(cmd, cwd=str(mech_dir), timeout=PLOT_TIMEOUT)
    return ret.returncode


def run_kl_divergence(
    *,
    mech_dir: Path,
    dataset: str,
    plot_output_name: str,
    seeds_for_plot: list[int],
    out_plot: str,
    output_inference_root: str = "output_inference",
) -> int:
    """Run plot_kl_divergence.py against the per-dataset plain/opinion dirs."""
    cmd = [
        sys.executable, "plot_kl_divergence.py",
        "--plain_dir", f"{output_inference_root}/{dataset}/plain",
        "--opinion_dir", f"{output_inference_root}/{dataset}/opinion_only",
        "--out_plot", out_plot,
        "--model_key", plot_output_name,
        "--data_seeds", *[str(s) for s in seeds_for_plot],
    ]
    print(f"[Plot] Running plot_kl_divergence.py -> {out_plot} ...")
    ret = subprocess.run(cmd, cwd=str(mech_dir), timeout=PLOT_TIMEOUT)
    return ret.returncode
