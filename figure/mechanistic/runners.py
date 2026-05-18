"""Mechanistic plotting runners: prepend mech sys.path then call numeric + figure render."""
from __future__ import annotations
import traceback
from pathlib import Path
from typing import Any

from figure.mechanistic.kl_plot import build_plot_kl_arg_parser, execute_plot_kl_divergence
from figure.mechanistic.mech_path import prepend_mech_syspath


def _resolved_inference_plain_opinion_dirs(
    mech_dir: Path,
    output_inference_root: str,
    dataset: str,
) -> tuple[str, str]:
    root = Path(output_inference_root)
    base = root.resolve() if root.is_absolute() else (mech_dir / root).resolve()
    plain = str(base / dataset / "plain")
    opinion = str(base / dataset / "opinion_only")
    return plain, opinion


def run_decision_score(
    *,
    mech_dir: Path,
    plain_paths: list[str],
    opinion_paths: list[str],
    out_plot: str,
) -> int:
    print(f"[Plot] compute_and_plot_decision_score -> {out_plot} ...")
    prepend_mech_syspath(mech_dir)
    try:
        from compute_decision_score import compute_and_plot_decision_score

        compute_and_plot_decision_score(
            plain_paths=plain_paths,
            opinion_paths=opinion_paths,
            out_plot=out_plot,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def run_kl_divergence(
    *,
    mech_dir: Path,
    dataset: str,
    plot_output_name: str,
    seeds_for_plot: list[int],
    out_plot: str,
    output_inference_root: str = "output_inference",
    **_: Any,
) -> int:
    plain_dir, opinion_dir = _resolved_inference_plain_opinion_dirs(mech_dir, output_inference_root, dataset)
    print(f"[Plot] execute_plot_kl_divergence -> {out_plot} ...")
    prepend_mech_syspath(mech_dir)
    try:
        parser = build_plot_kl_arg_parser()
        args = parser.parse_args([])
        args.plain_dir = plain_dir
        args.opinion_dir = opinion_dir
        args.out_plot = out_plot
        args.model_key = plot_output_name
        args.data_seeds = list(seeds_for_plot)
        args.data_seed = None
        execute_plot_kl_divergence(args)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
