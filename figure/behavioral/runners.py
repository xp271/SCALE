"""Pipeline entrypoints: behavioral Fig1 / Fig2."""
from __future__ import annotations

import os
import traceback
from types import SimpleNamespace
from typing import Any

from figure.behavioral.figures import plot_behavioral_figures


def run_fig1(
    *,
    dataset: str,
    figure_dir: str,
    plot_output_name: str,
    plot_name_with_dataset: str,
    seeds_for_plot: list[int],
    eval_sr_correct_only: bool,
    output_base: str = "output",
    baseline_model_type: str | None = None,
    **_: Any,
) -> int:
    baseline = baseline_model_type or (
        f"{plot_output_name.split('_', 1)[0]}_full_precision" if eval_sr_correct_only else None
    )
    print(f"[Plot] plot_behavioral(fig1) for {plot_name_with_dataset} (avg over seeds {seeds_for_plot}) ...")
    ns = SimpleNamespace(
        model_type=plot_output_name,
        which="fig1",
        figure_dir=figure_dir,
        output_base=os.path.abspath(str(output_base)),
        dataset_subdir=dataset,
        data_seed=None,
        data_seeds=list(seeds_for_plot),
        output_suffix=plot_name_with_dataset,
        correct_only_sr=eval_sr_correct_only,
        baseline_model_type=baseline,
    )
    try:
        plot_behavioral_figures(ns)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def run_sycophancy_summary(
    *,
    dataset: str,
    figure_dir: str,
    plot_model_id: str,
    plot_method_id: str,
    plot_name_with_dataset: str,
    seeds_for_plot: list[int],
    output_base: str,
    with_authority: bool,
    **_: Any,
) -> int:
    """OSR / ASR (mean ± std over seeds) and Delta vs. full precision for one quantized combo."""
    from figure.behavioral.delta_sr import (
        FP_METHOD_ID,
        collect_rates,
        print_summary,
        summarize_against_fp,
        write_summary_csv,
    )

    if plot_method_id == FP_METHOD_ID:
        return 0
    print(f"[Plot] OSR/ASR summary for {plot_name_with_dataset} (seeds {seeds_for_plot}) ...")
    try:
        base = os.path.abspath(str(output_base))
        rates_fp = collect_rates(base, dataset, plot_model_id, FP_METHOD_ID, seeds_for_plot, with_authority=with_authority)
        rates_q = collect_rates(base, dataset, plot_model_id, plot_method_id, seeds_for_plot, with_authority=with_authority)
        summary = summarize_against_fp(rates_q, rates_fp)
        print_summary(plot_method_id, summary)
        method, _, bits = plot_method_id.rpartition("_w")
        rows = [
            {"method_id": plot_method_id, "method": method, "bits": bits, "metric": m, **s}
            for m, s in summary.items()
        ]
        write_summary_csv(rows, os.path.join(figure_dir, f"sycophancy_{plot_name_with_dataset}.csv"))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def run_fig2_authority(
    *,
    dataset: str,
    figure_dir: str,
    plot_output_name: str,
    plot_name_with_dataset: str,
    seeds_for_plot: list[int],
    output_base: str = "output",
    **_: Any,
) -> int:
    print(
        f"[Plot] plot_behavioral(fig2 advanced) for {plot_name_with_dataset} "
        f"(avg over seeds {seeds_for_plot}) ..."
    )
    ns = SimpleNamespace(
        model_type=plot_output_name,
        which="fig2",
        figure_dir=figure_dir,
        output_base=os.path.abspath(str(output_base)),
        dataset_subdir=dataset,
        data_seed=None,
        data_seeds=list(seeds_for_plot),
        output_suffix=plot_name_with_dataset,
        correct_only_sr=False,
        baseline_model_type=None,
    )
    try:
        plot_behavioral_figures(ns)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
