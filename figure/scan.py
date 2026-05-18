"""Scan existing output/{dataset}/plain pkl files to enumerate plottable methods.

Used by the ``--plot_scan_existing`` mode: given a save root (or LLM-sycophancy
behavioral_analysis subdir) and a ``model_id_fs``, list every ``method_id``
whose both ``plain`` and ``opinion_only`` pkl already exist for the requested
seeds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from figure.behavioral import run_fig1
from utils.paths import DIR_SYCO_SCRIPT


def split_method_and_seed_from_stem(stem: str, model_id_fs: str) -> tuple[str | None, int | None]:
    """Parse method_id and seed from stem of {model_id_fs}_{method_id}[_seed].pkl."""
    prefix = f"{model_id_fs}_"
    if not stem.startswith(prefix):
        return None, None
    body = stem[len(prefix) :]
    m = re.match(r"^(.*)_(\d+)$", body)
    if m:
        return m.group(1), int(m.group(2))
    return body, None


def scan_existing_methods_for_plot(
    plain_dir: Path,
    opinion_dir: Path,
    model_id_fs: str,
    seeds: list[int] | None,
) -> list[str]:
    """Scan plain_dir pkls; return method_ids that also have a match in opinion_dir."""
    if not plain_dir.exists():
        return []
    methods = set()
    for p in plain_dir.glob(f"{model_id_fs}_*.pkl"):
        method_id, seed = split_method_and_seed_from_stem(p.stem, model_id_fs)
        if not method_id:
            continue
        if seeds is not None and seed not in seeds:
            continue
        op_path = opinion_dir / p.name
        if not op_path.exists():
            continue
        methods.add(method_id)
    return sorted(methods)


def run_plot_from_existing_pkls(
    *,
    syco_repo: Path,
    dataset: str,
    model_id_fs: str,
    seeds_for_plot: list[int],
    figure_dir: str = "figure",
    correct_only_sr: bool = True,
    behavioral_output_base: str | None = None,
) -> int:
    """Batch plot from existing pkls only (no quant/eval).

    When ``behavioral_output_base`` is None, scan legacy layout
    ``LLM-sycophancy/experiments/behavioral_analysis/output/{dataset}/...``；
    When set (save_root/behavioral), use it as scan root.
    """
    ba_dir = syco_repo / DIR_SYCO_SCRIPT
    if behavioral_output_base:
        base = Path(behavioral_output_base).resolve()
    else:
        base = (ba_dir / "output").resolve()
    plain_dir = base / dataset / "plain"
    opinion_dir = base / dataset / "opinion_only"
    methods = scan_existing_methods_for_plot(plain_dir, opinion_dir, model_id_fs, seeds_for_plot if seeds_for_plot else None)
    if not methods:
        print(
            f"[PlotOnly] no eligible methods found in {plain_dir} for model_id_fs={model_id_fs}",
            file=sys.stderr,
        )
        return 0
    ok_cnt = 0
    out_base_arg = str(base)
    for method_id in methods:
        plot_output_name = f"{model_id_fs}_{method_id}"
        out_suffix = f"{dataset}_{plot_output_name}" + ("_correct_only" if correct_only_sr else "")
        print(f"[PlotOnly] plotting {plot_output_name} (seeds={seeds_for_plot}) ...")
        rc = run_fig1(
            dataset=dataset,
            figure_dir=figure_dir,
            plot_output_name=plot_output_name,
            plot_name_with_dataset=out_suffix,
            seeds_for_plot=seeds_for_plot,
            eval_sr_correct_only=correct_only_sr,
            output_base=out_base_arg,
            baseline_model_type=f"{model_id_fs}_full_precision" if correct_only_sr else None,
        )
        if rc == 0:
            ok_cnt += 1
    print(f"[PlotOnly] done: {ok_cnt}/{len(methods)} methods plotted.")
    return ok_cnt
