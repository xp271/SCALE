"""OSR / ASR and their relative change by quantization (paper Eq. 1 / Eq. 2, Figures 1-2, Tables 5-7).

CLI (repo root): ``python -m figure.behavioral.delta_sr --output_base save/behavioral --dataset mmlu
--model_id llama_3.1_8b_instruct``. Writes a CSV with mean ± std over seeds for every
(method, bit-width, metric) and, unless ``--no_plot``, the Delta_OSR / Delta_ASR figures.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

from figure.behavioral.constants import LEVELS_ACADEMIC, output_dirs_for_dataset
from figure.behavioral.metrics import (
    authority_sycophancy_rate,
    mean_std,
    opinion_sycophancy_rate,
    relative_change,
)

FP_METHOD_ID = "full_precision"
DEFAULT_METHODS = ("rtn", "gptq", "awq", "hqq")
DEFAULT_BITS = (8, 6, 4)
OSR_METRIC = "OSR"


def asr_metric(level: str) -> str:
    return f"ASR_{level}"


def _pkl_path(output_base: str, subdir: str, model_id: str, method_id: str, seed: int) -> str:
    return os.path.join(output_base, subdir, f"{model_id}_{method_id}_{seed}.pkl")


def collect_rates(
    output_base: str,
    dataset: str,
    model_id: str,
    method_id: str,
    seeds: list[int],
    *,
    with_authority: bool = True,
) -> dict[str, dict[int, float]]:
    """Per-seed OSR and ASR^L (%) of one model variant: {metric: {seed: rate}}. Missing pkls are skipped."""
    dirs_op, dirs_auth = output_dirs_for_dataset(dataset)
    rates: dict[str, dict[int, float]] = {OSR_METRIC: {}}
    if with_authority:
        rates.update({asr_metric(level): {} for level in LEVELS_ACADEMIC})
    for seed in seeds:
        op_path = _pkl_path(output_base, dirs_op["opinion_only"], model_id, method_id, seed)
        if not os.path.isfile(op_path):
            print(f"  [skip] missing {op_path}")
            continue
        df_op = pd.read_pickle(op_path)
        osr = opinion_sycophancy_rate(df_op)
        if osr is not None:
            rates[OSR_METRIC][seed] = osr
        if not with_authority:
            continue
        for level in LEVELS_ACADEMIC:
            au_path = _pkl_path(output_base, dirs_auth[level], model_id, method_id, seed)
            if not os.path.isfile(au_path):
                print(f"  [skip] missing {au_path}")
                continue
            asr = authority_sycophancy_rate(df_op, pd.read_pickle(au_path))
            if asr is not None:
                rates[asr_metric(level)][seed] = asr
    return rates


def summarize_against_fp(
    rates_q: dict[str, dict[int, float]],
    rates_fp: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """For each metric: mean/std over seeds and the relative change vs. FP.

    ``delta`` uses the seed-averaged rates; ``delta_std`` is the std of the per-seed paired deltas.
    """
    out: dict[str, dict[str, float]] = {}
    for metric, by_seed in rates_q.items():
        fp_by_seed = rates_fp.get(metric, {})
        mean_q, std_q = mean_std(list(by_seed.values()))
        mean_fp, std_fp = mean_std(list(fp_by_seed.values()))
        paired = [relative_change(by_seed[s], fp_by_seed[s]) for s in by_seed if s in fp_by_seed]
        _, delta_std = mean_std(paired)
        out[metric] = {
            "mean": mean_q,
            "std": std_q,
            "n_seeds": len(by_seed),
            "fp_mean": mean_fp,
            "fp_std": std_fp,
            "delta": relative_change(mean_q, mean_fp) if by_seed and fp_by_seed else float("nan"),
            "delta_std": delta_std,
        }
    return out


def _fmt(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.2f}"


def write_summary_csv(rows: list[dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    keys = ["method_id", "method", "bits", "metric", "mean", "std", "n_seeds", "fp_mean", "fp_std", "delta", "delta_std"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"Saved: {out_csv}")


def print_summary(label: str, summary: dict[str, dict[str, float]]) -> None:
    for metric, s in summary.items():
        print(
            f"  {label} {metric}: {_fmt(s['mean'])} +/- {_fmt(s['std'])} (n={s['n_seeds']}) | "
            f"FP {_fmt(s['fp_mean'])} +/- {_fmt(s['fp_std'])} | Delta = {_fmt(s['delta'])}% +/- {_fmt(s['delta_std'])}"
        )


def run_delta_sr(args: argparse.Namespace) -> list[dict]:
    seeds = resolve_seeds(args)
    output_base = os.path.abspath(args.output_base)
    with_authority = not args.no_authority
    print(f"[delta_sr] {args.model_id} on {args.dataset}, seeds={seeds}")
    rates_fp = collect_rates(output_base, args.dataset, args.model_id, FP_METHOD_ID, seeds, with_authority=with_authority)

    rows: list[dict] = []
    # FP rows carry the absolute rates only (Delta is 0 by definition)
    for metric, by_seed in rates_fp.items():
        m, s = mean_std(list(by_seed.values()))
        rows.append({"method_id": FP_METHOD_ID, "method": "fp", "bits": 16, "metric": metric,
                     "mean": m, "std": s, "n_seeds": len(by_seed), "fp_mean": m, "fp_std": s,
                     "delta": 0.0 if by_seed else float("nan"), "delta_std": 0.0 if by_seed else float("nan")})
    deltas: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for method in args.methods:
        for bits in args.bits:
            method_id = f"{method}_w{bits}"
            rates_q = collect_rates(output_base, args.dataset, args.model_id, method_id, seeds, with_authority=with_authority)
            summary = summarize_against_fp(rates_q, rates_fp)
            deltas[(method, bits)] = summary
            print_summary(method_id, summary)
            for metric, s in summary.items():
                rows.append({"method_id": method_id, "method": method, "bits": bits, "metric": metric, **s})

    figure_dir = args.figure_dir
    out_csv = args.out_csv or os.path.join(figure_dir, f"sycophancy_rates_{args.dataset}_{args.model_id}.csv")
    write_summary_csv(rows, out_csv)

    if not args.no_plot:
        from figure.behavioral.render_delta import save_authority_delta_figure, save_legends, save_opinion_delta_figure

        label = args.model_label or args.model_id
        osr_series = {
            m: [deltas[(m, b)][OSR_METRIC]["delta"] for b in args.bits] for m in args.methods
        }
        save_opinion_delta_figure(
            osr_series,
            bits=list(args.bits),
            out_plot=os.path.join(figure_dir, f"opinion_delta_sr_{label}_{args.dataset}.pdf"),
            ylim=tuple(args.osr_ylim) if args.osr_ylim else None,
        )
        if with_authority:
            for m in args.methods:
                asr_series = {
                    level: [deltas[(m, b)][asr_metric(level)]["delta"] for b in args.bits] for level in LEVELS_ACADEMIC
                }
                save_authority_delta_figure(
                    asr_series,
                    bits=list(args.bits),
                    out_plot=os.path.join(figure_dir, f"authority_delta_sr_{m}_{label}_{args.dataset}.pdf"),
                    ylim=tuple(args.asr_ylim) if args.asr_ylim else None,
                )
        save_legends(figure_dir, methods=list(args.methods), with_authority=with_authority)
    return rows


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.data_seeds:
        return list(args.data_seeds)
    from utils.seeds import generate_eval_data_seeds

    return generate_eval_data_seeds(args.data_seed_rng, args.eval_avg_runs)


def build_delta_sr_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OSR / ASR (mean +/- std over seeds) and Delta_OSR / Delta_ASR vs. full precision")
    p.add_argument("--output_base", required=True, help="Behavioral pkl root, e.g. save/behavioral")
    p.add_argument("--dataset", required=True, help="Dataset subdir, e.g. mmlu / commonsenseqa")
    p.add_argument("--model_id", required=True, help="model_id as in pkl filenames, e.g. llama_3.1_8b_instruct")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Method ids without bit suffix")
    p.add_argument("--bits", nargs="+", type=int, default=list(DEFAULT_BITS), help="Bit-widths, plotted left to right")
    p.add_argument("--data_seeds", nargs="+", type=int, default=None, help="Explicit seeds; default: from --data_seed_rng")
    p.add_argument("--data_seed_rng", type=int, default=42, help="Same as run_pipeline.py")
    p.add_argument("--eval_avg_runs", type=int, default=3, help="Same as run_pipeline.py")
    p.add_argument("--no_authority", action="store_true", help="Only OSR (skip ASR)")
    p.add_argument("--figure_dir", default="result/delta_sr", help="Output directory for CSV and figures")
    p.add_argument("--out_csv", default="", help="CSV path (default: <figure_dir>/sycophancy_rates_<dataset>_<model_id>.csv)")
    p.add_argument("--model_label", default="", help="Short model name in figure filenames, e.g. llama3.1")
    p.add_argument("--osr_ylim", nargs=2, type=float, default=None, help="y-range of the Delta_OSR plot")
    p.add_argument("--asr_ylim", nargs=2, type=float, default=None, help="y-range of the Delta_ASR plots (default -5 5)")
    p.add_argument("--no_plot", action="store_true", help="Only write the CSV")
    return p


def main_cli() -> None:
    run_delta_sr(build_delta_sr_arg_parser().parse_args())


if __name__ == "__main__":
    main_cli()
