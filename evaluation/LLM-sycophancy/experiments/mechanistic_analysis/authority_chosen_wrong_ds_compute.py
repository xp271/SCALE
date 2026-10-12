"""Authority-level chosen_wrong DS aggregation (compute-only), averaged over data seeds. Plotting: ``authority_ds_render``.

For each authority level and seed, the per-layer mean DS of the user's (incorrect) belief is computed
over all questions; the curve of a level is then the mean of its per-seed curves (each seed weighted
equally). ``--out_csv`` additionally stores the per-layer std across seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compute_decision_score import decision_score
from mcq_option_utils import contiguous_letters_from_logits, is_single_upper_option_letter

LEVELS = ("beginner", "intermediate", "advanced")


def compute_wrong_ds_per_layer(layer_logits: dict, wrong_letter: str):
    if not isinstance(layer_logits, dict):
        return []
    out = []
    for key, logits in layer_logits.items():
        if not isinstance(logits, dict) or contiguous_letters_from_logits(logits) is None:
            continue
        if not is_single_upper_option_letter(wrong_letter, logits):
            continue
        try:
            layer_idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        ds_wrong = decision_score(logits.get(wrong_letter, 0.0), logits)
        out.append((layer_idx, ds_wrong))
    return sorted(out, key=lambda x: x[0])


def load_chosen_wrong_rows(pkl_path: Path):
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"read failed, skipping: {pkl_path} ({e})")
        return []

    if "layer_logits" not in df.columns or "chosen_wrong_answer_index" not in df.columns:
        print(f"missing columns, skipping: {pkl_path}")
        return []

    rows = []
    for _, row in df.iterrows():
        layer_logits = row.get("layer_logits")
        wrong_letter = row.get("chosen_wrong_answer_index")
        if not isinstance(layer_logits, dict) or pd.isna(wrong_letter):
            continue
        wrong_letter = str(wrong_letter).strip().upper()
        if len(wrong_letter) != 1:
            continue
        for layer_idx, ds_wrong in compute_wrong_ds_per_layer(layer_logits, wrong_letter):
            rows.append({"layer": layer_idx, "ds_chosen_wrong": ds_wrong})
    return rows


def authority_logit_pkl(root: Path, dataset: str, level: str, model_output_name: str, data_seed: int) -> Path:
    return (
        root
        / dataset
        / "prefix_and_opinion"
        / "academic"
        / "original"
        / level
        / f"{model_output_name}_logit_all_{data_seed}.pkl"
    )


def resolve_data_seeds(args: argparse.Namespace) -> list[int]:
    seeds = getattr(args, "data_seeds", None)
    return list(seeds) if seeds else [args.data_seed]


def build_default_authority_out_plot(dataset: str, model_output_name: str, data_seeds: list[int]) -> str:
    suffix = f"_{data_seeds[0]}" if len(data_seeds) == 1 else ""
    return f"figure/ds_authority_{dataset}_{model_output_name}{suffix}.png"


def collect_authority_ds_curves(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], str] | None:
    root = Path(args.output_inference_root).resolve()
    seeds = resolve_data_seeds(args)
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    csv_rows: list[dict] = []

    for level in LEVELS:
        per_seed: list[pd.Series] = []
        for seed in seeds:
            pkl_path = authority_logit_pkl(root, args.dataset, level, args.model_output_name, seed)
            if not pkl_path.exists():
                print(f"file missing, skipping {level} seed={seed}: {pkl_path}")
                continue
            rows = load_chosen_wrong_rows(pkl_path)
            if not rows:
                print(f"no usable data, skipping {level} seed={seed}: {pkl_path}")
                continue
            per_seed.append(pd.DataFrame(rows).groupby("layer")["ds_chosen_wrong"].mean())
        if not per_seed:
            continue
        if len(per_seed) < len(seeds):
            print(f"warning: {level} averaged over {len(per_seed)}/{len(seeds)} seeds")
        by_seed = pd.concat(per_seed, axis=1).sort_index()
        mean = by_seed.mean(axis=1)
        std = by_seed.std(axis=1, ddof=0)
        curves.append((level, mean.index.to_numpy(), mean.to_numpy()))
        for layer in mean.index:
            csv_rows.append({"level": level, "layer": int(layer), "ds_mean": mean[layer], "ds_std": std[layer], "n_seeds": len(per_seed)})

    if not curves:
        print("no curves to plot.")
        return None

    out_csv = getattr(args, "out_csv", "")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    out_plot = args.out_plot.strip() or build_default_authority_out_plot(args.dataset, args.model_output_name, seeds)
    return curves, out_plot


def build_authority_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="authority three-level chosen_wrong DS (averaged over seeds)")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--model_output_name", type=str, default="llama_3.1_8b_instruct_rtn_w4")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=None, help="Seeds to average over")
    parser.add_argument("--data_seed", type=int, default=42, help="Single seed (used when --data_seeds is not set)")
    parser.add_argument("--out_plot", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="", help="Optional: per-layer mean / std across seeds")
    return parser
