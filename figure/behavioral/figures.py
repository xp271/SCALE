"""Aggregate behavioral figure data and trigger plotting."""
from __future__ import annotations

import argparse
import os
from typing import Any

from figure.behavioral.constants import LABELS_ACADEMIC_LEVEL, output_dirs_for_dataset
from figure.behavioral.metrics import (
    avg_metrics,
    find_first_pkl,
    get_correct_only_metrics,
    get_metrics,
    pkl_to_model_label,
)
from figure.behavioral.render_behavioral import save_figure1, save_figure2


def plot_behavioral_figures(args: Any) -> None:
    """Args equivalent to legacy plot_figure2 CLI (Namespace/SimpleNamespace)."""
    draw_fig1 = args.which in ("both", "fig1")
    draw_fig2 = args.which in ("both", "fig2")
    data_seed = args.data_seed
    data_seeds = args.data_seeds
    output_suffix = args.output_suffix
    correct_only_sr = bool(args.correct_only_sr)
    baseline_model_type = args.baseline_model_type
    if data_seeds is not None and data_seed is not None:
        data_seed = None
    if correct_only_sr and draw_fig1 and not baseline_model_type:
        raise ValueError("baseline_model_type required when correct_only_sr is enabled")

    base = os.path.abspath(str(args.output_base))
    dirs_fig1, dirs_fig2 = output_dirs_for_dataset(args.dataset_subdir)

    def _collect_metrics_over_seeds(seeds: list[int], draw_f1: bool, draw_f2: bool):
        plain_list, opinion_list = [], []
        fig2_list = {k: [] for k in dirs_fig2}
        for s in seeds:
            p_plain = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), args.model_type, s) if draw_f1 else None
            p_op = find_first_pkl(os.path.join(base, dirs_fig1["opinion_only"]), args.model_type, s) if draw_f1 else None
            if draw_f1 and p_plain and p_op:
                m_plain, m_op = get_metrics(p_plain), get_metrics(p_op)
                if m_plain and m_op:
                    plain_list.append(m_plain)
                    opinion_list.append(m_op)
            if draw_f2:
                for level, subdir in dirs_fig2.items():
                    pkl_path = find_first_pkl(os.path.join(base, subdir), args.model_type, s)
                    if pkl_path:
                        m = get_metrics(pkl_path)
                        if m:
                            fig2_list[level].append(m)
        return plain_list, opinion_list, fig2_list

    plain: dict[str, float] | None = None
    opinion: dict[str, float] | None = None
    opinion_correct_only: dict[str, float] | None = None
    fig2_metrics: list[dict[str, float]] = []
    model_label = "Model"

    if data_seeds:
        plain_list, opinion_list, fig2_list = _collect_metrics_over_seeds(data_seeds, draw_fig1, draw_fig2)
        if draw_fig1 and (not plain_list or not opinion_list):
            raise FileNotFoundError(f"insufficient fig1 data (seeds={data_seeds})")
        if draw_fig2 and not all(len(fig2_list[k]) > 0 for k in dirs_fig2):
            raise FileNotFoundError(f"insufficient fig2 data (seeds={data_seeds})")
        plain = avg_metrics(plain_list) if draw_fig1 else None
        opinion = avg_metrics(opinion_list) if draw_fig1 else None
        if draw_fig1 and correct_only_sr:
            corr_list = []
            for s in data_seeds:
                p_cur_plain = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), args.model_type, s)
                p_base_plain = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), baseline_model_type, s)
                p_op = find_first_pkl(os.path.join(base, dirs_fig1["opinion_only"]), args.model_type, s)
                if not (p_cur_plain and p_base_plain and p_op):
                    continue
                m_corr = get_correct_only_metrics(p_op, p_cur_plain, p_base_plain)
                if m_corr:
                    corr_list.append(m_corr)
            opinion_correct_only = avg_metrics(corr_list)
        fig2_metrics = []
        if draw_fig2:
            for level in LABELS_ACADEMIC_LEVEL:
                fig2_metrics.append(avg_metrics(fig2_list[level.lower()]))
        first_pkl = None
        for s in data_seeds:
            fp = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), args.model_type, s) if draw_fig1 else None
            if fp:
                first_pkl = fp
                break
        model_label = pkl_to_model_label(first_pkl) if first_pkl else "Model"
    else:
        plain_pkl = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), args.model_type, data_seed) if draw_fig1 else None
        opinion_pkl = find_first_pkl(os.path.join(base, dirs_fig1["opinion_only"]), args.model_type, data_seed) if draw_fig1 else None
        fig2_pkls = {}
        if draw_fig2:
            for level, subdir in dirs_fig2.items():
                pkl_path = find_first_pkl(os.path.join(base, subdir), args.model_type, data_seed)
                if not pkl_path:
                    raise FileNotFoundError(
                        f"missing {level} pkl (dir {os.path.join(base, subdir)}"
                        + (f", model_type={args.model_type}" if args.model_type else "")
                        + "）"
                    )
                fig2_pkls[level] = pkl_path
        first_pkl = plain_pkl or opinion_pkl or (list(fig2_pkls.values())[0] if fig2_pkls else None)
        model_label = pkl_to_model_label(first_pkl)
        plain = get_metrics(plain_pkl) if plain_pkl else None
        opinion = get_metrics(opinion_pkl) if opinion_pkl else None
        opinion_correct_only = None
        if draw_fig1 and correct_only_sr:
            base_plain_pkl = find_first_pkl(os.path.join(base, dirs_fig1["plain"]), baseline_model_type, data_seed)
            if opinion_pkl and plain_pkl and base_plain_pkl:
                opinion_correct_only = get_correct_only_metrics(opinion_pkl, plain_pkl, base_plain_pkl)
        fig2_metrics = []
        if draw_fig2:
            for level in LABELS_ACADEMIC_LEVEL:
                m = get_metrics(fig2_pkls[level.lower()])
                if m is None:
                    raise FileNotFoundError(f"cannot compute metrics from {fig2_pkls[level.lower()]}")
                fig2_metrics.append(m)

    if draw_fig1:
        if correct_only_sr:
            if opinion_correct_only is None:
                raise FileNotFoundError("cannot compute correct_only SR (check plain/opinion pkls and baseline match)")
        else:
            if plain is None:
                raise FileNotFoundError("cannot compute plain metrics (missing columns or no valid predictions)")
            if opinion is None:
                raise FileNotFoundError("cannot compute opinion metrics")
    if draw_fig2 and not data_seeds and len(fig2_metrics) == 3:
        same = fig2_metrics[0] == fig2_metrics[1] == fig2_metrics[2]
        if same:
            print("  [warning] Beginner / Intermediate / Advanced metrics are identical; confirm pkls under output dirs came from different inputs.")

    if draw_fig1:
        save_figure1(
            figure_dir=args.figure_dir,
            correct_only_sr=correct_only_sr,
            plain=plain,
            opinion=opinion,
            opinion_correct_only=opinion_correct_only,
            model_label=model_label,
            output_suffix=output_suffix,
            data_seed=data_seed,
            data_seeds=list(data_seeds) if data_seeds else None,
        )

    if draw_fig2:
        save_figure2(
            figure_dir=args.figure_dir,
            fig2_metrics=fig2_metrics,
            model_label=model_label,
            output_suffix=output_suffix,
            data_seed=data_seed,
            data_seeds=list(data_seeds) if data_seeds else None,
        )


def build_behavioral_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Figure-2 style charts (Plain vs Opinion; Advanced First-pov)")
    parser.add_argument("--model_type", type=str, default=None, help="Model keyword: partial match on pkl filename (case-insensitive)")
    parser.add_argument("--which", type=str, default="both", choices=["both", "fig1", "fig2"], help="Which figure to plot")
    parser.add_argument("--figure_dir", type=str, default="figure", help="Figure output directory")
    parser.add_argument("--output_base", type=str, default="output", help="output root directory")
    parser.add_argument("--dataset_subdir", type=str, default="mmlu", help="First-level subdir under output/")
    parser.add_argument("--data_seed", type=int, default=None, help="Single seed")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=None, help="Average over multiple seeds")
    parser.add_argument("--output_suffix", type=str, default=None, help="Output filename suffix")
    parser.add_argument("--correct_only_sr", action="store_true", help="correct-only SR mode")
    parser.add_argument("--baseline_model_type", type=str, default=None, help="Baseline model keyword (correct_only)")
    return parser


def main_cli() -> None:
    parser = build_behavioral_arg_parser()
    args = parser.parse_args()
    plot_behavioral_figures(args)
