"""Plot phase orchestration: per (model, method) combo run fig1/fig2/DS/KL."""
from __future__ import annotations

import sys
from pathlib import Path

from evaluation.paths import expected_syco_pkl_path
from figure.behavioral import run_fig1, run_fig2_authority
from figure.mechanistic import run_decision_score, run_kl_divergence

from utils.paths import DIR_SYCO_SCRIPT

MECH_SUBDIR = "experiments/mechanistic_analysis"


def _resolve_behavioral_output_base(syco_repo: Path, behavioral_output_base: str) -> str:
    """Relative paths like ``output`` match legacy cwd=behavioral_analysis behavior."""
    p = Path(behavioral_output_base)
    if p.is_absolute():
        return str(p.resolve())
    return str((syco_repo / DIR_SYCO_SCRIPT / p).resolve())


def plot_combo(
    *,
    syco_repo: Path,
    plot_model_id: str,
    plot_method_id: str,
    dataset: str,
    seeds_for_plot: list[int],
    figure_dir: str,
    eval_authority_advanced: bool,
    eval_mechanistic: bool,
    eval_sr_correct_only: bool,
    eval_jobs: list[dict],
    behavioral_output_base: str = "output",
    mechanistic_output_base: str = "output_inference",
) -> None:
    """Generate every figure that the legacy pipeline produced for one (model, method)."""
    mech_dir = syco_repo / MECH_SUBDIR
    behavioral_resolved = _resolve_behavioral_output_base(syco_repo, behavioral_output_base)
    plot_output_name = f"{plot_model_id}_{plot_method_id}"
    plot_name_with_dataset = f"{dataset}_{plot_output_name}"
    if eval_sr_correct_only:
        plot_name_with_dataset = f"{plot_name_with_dataset}_correct_only"

    # 1) fig1: Plain vs Opinion-Only
    run_fig1(
        dataset=dataset,
        figure_dir=figure_dir,
        plot_output_name=plot_output_name,
        plot_name_with_dataset=plot_name_with_dataset,
        seeds_for_plot=seeds_for_plot,
        eval_sr_correct_only=eval_sr_correct_only,
        output_base=behavioral_resolved,
        baseline_model_type=f"{plot_model_id}_full_precision",
    )

    # 1.1) fig2: First-pov Academic (eval_authority_advanced only)
    if eval_authority_advanced:
        run_fig2_authority(
            dataset=dataset,
            figure_dir=figure_dir,
            plot_output_name=plot_output_name,
            plot_name_with_dataset=plot_name_with_dataset,
            seeds_for_plot=seeds_for_plot,
            output_base=behavioral_resolved,
        )

    # 2) compute_decision_score: mechanistic branch only
    logit_plain_job = next((j for j in eval_jobs if j.get("tag") == "logit_cot_plain"), None)
    logit_opinion_job = next((j for j in eval_jobs if j.get("tag") == "logit_cot_opinion"), None)
    if eval_mechanistic and logit_plain_job and logit_opinion_job:
        dummy_path = Path("dummy")
        plain_paths: list[str] = []
        opinion_paths: list[str] = []
        for s in seeds_for_plot:
            pp = expected_syco_pkl_path(
                mech_dir, logit_plain_job, dummy_path, dataset,
                data_seed=s, model_output_name=plot_output_name,
                output_base_override=mechanistic_output_base,
            )
            op = expected_syco_pkl_path(
                mech_dir, logit_opinion_job, dummy_path, dataset,
                data_seed=s, model_output_name=plot_output_name,
                output_base_override=mechanistic_output_base,
            )
            if pp.exists():
                plain_paths.append(str(pp))
            if op.exists():
                opinion_paths.append(str(op))
        if len(plain_paths) == len(seeds_for_plot) and len(opinion_paths) == len(seeds_for_plot):
            ds_out = str(Path(figure_dir) / f"ds_{plot_name_with_dataset}.png")
            run_decision_score(
                mech_dir=mech_dir,
                plain_paths=plain_paths,
                opinion_paths=opinion_paths,
                out_plot=ds_out,
            )
        else:
            print(
                f"[Plot] Skip compute_decision_score ({plot_name_with_dataset}): missing pkl for some seeds",
                file=sys.stderr,
            )
    elif not eval_mechanistic:
        print("[Plot] Skip compute_decision_score: eval_mechanistic=false", file=sys.stderr)
    else:
        print("[Plot] Skip compute_decision_score: logit_cot jobs not found", file=sys.stderr)

    # 3) plot_kl_divergence: average over seeds (mechanistic)
    if eval_mechanistic:
        kl_out = str(Path(figure_dir) / f"kl_divergence_{plot_name_with_dataset}.png")
        run_kl_divergence(
            mech_dir=mech_dir,
            dataset=dataset,
            plot_output_name=plot_output_name,
            seeds_for_plot=seeds_for_plot,
            out_plot=kl_out,
            output_inference_root=mechanistic_output_base,
        )
    else:
        print("[Plot] Skip plot_kl_divergence: eval_mechanistic=false", file=sys.stderr)

    print(f"[Plot] Done {plot_name_with_dataset}")


def run_plot_phase(
    *,
    syco_repo: Path,
    completed_combos: list[tuple[str, str]],
    dataset: str,
    seeds_for_plot: list[int],
    figure_dir: str,
    eval_authority_advanced: bool,
    eval_mechanistic: bool,
    eval_sr_correct_only: bool,
    eval_jobs: list[dict],
    behavioral_output_base: str = "output",
    mechanistic_output_base: str = "output_inference",
) -> None:
    """Loop over every completed (model, method) producing the four figure types."""
    if not seeds_for_plot:
        seeds_for_plot = [42]
    for plot_model_id, plot_method_id in completed_combos:
        plot_combo(
            syco_repo=syco_repo,
            plot_model_id=plot_model_id,
            plot_method_id=plot_method_id,
            dataset=dataset,
            seeds_for_plot=seeds_for_plot,
            figure_dir=figure_dir,
            eval_authority_advanced=eval_authority_advanced,
            eval_mechanistic=eval_mechanistic,
            eval_sr_correct_only=eval_sr_correct_only,
            eval_jobs=eval_jobs,
            behavioral_output_base=behavioral_output_base,
            mechanistic_output_base=mechanistic_output_base,
        )
    print(f"Plot phase done: all combos, 1 figure per combo (avg over seeds) {seeds_for_plot}")
