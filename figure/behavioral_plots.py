"""Behavioral plots: plot_figure2.py for fig1 (Plain vs Opinion-Only) and fig2 (Authority)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PLOT_TIMEOUT = 300


def run_fig1(
    *,
    ba_dir: Path,
    dataset: str,
    figure_dir: str,
    plot_output_name: str,
    plot_name_with_dataset: str,
    seeds_for_plot: list[int],
    eval_sr_correct_only: bool,
    output_base: str = "output",
    baseline_model_type: str | None = None,
) -> int:
    """Run plot_figure2.py --which fig1 (Plain vs Opinion-Only)."""
    cmd = [
        sys.executable, "plot_figure2.py",
        "--which", "fig1",
        "--output_base", str(output_base),
        "--dataset_subdir", dataset,
        "--figure_dir", figure_dir,
        "--model_type", plot_output_name,
        "--output_suffix", plot_name_with_dataset,
        "--data_seeds", *[str(s) for s in seeds_for_plot],
    ]
    if eval_sr_correct_only:
        cmd.extend(
            [
                "--correct_only_sr",
                "--baseline_model_type",
                baseline_model_type or f"{plot_output_name.split('_', 1)[0]}_full_precision",
            ]
        )
    print(
        f"[Plot] Running plot_figure2.py(fig1) for {plot_name_with_dataset} "
        f"(avg over seeds {seeds_for_plot}) ..."
    )
    ret = subprocess.run(cmd, cwd=str(ba_dir), timeout=PLOT_TIMEOUT)
    return ret.returncode


def run_fig2_authority(
    *,
    ba_dir: Path,
    dataset: str,
    figure_dir: str,
    plot_output_name: str,
    plot_name_with_dataset: str,
    seeds_for_plot: list[int],
    output_base: str = "output",
) -> int:
    """Run plot_figure2.py --which fig2 (First-pov Academic Beginner/Intermediate/Advanced)."""
    cmd = [
        sys.executable, "plot_figure2.py",
        "--which", "fig2",
        "--output_base", str(output_base),
        "--dataset_subdir", dataset,
        "--figure_dir", figure_dir,
        "--model_type", plot_output_name,
        "--output_suffix", plot_name_with_dataset,
        "--data_seeds", *[str(s) for s in seeds_for_plot],
    ]
    print(
        f"[Plot] Running plot_figure2.py(fig2 advanced) for {plot_name_with_dataset} "
        f"(avg over seeds {seeds_for_plot}) ..."
    )
    ret = subprocess.run(cmd, cwd=str(ba_dir), timeout=PLOT_TIMEOUT)
    return ret.returncode
