"""Authority-level chosen_wrong DS aggregation (compute-only). Plotting: ``authority_ds_render``."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compute_decision_score import decision_score
from mcq_option_utils import contiguous_letters_from_logits, is_single_upper_option_letter


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
        if pd.isna(layer_logits) or pd.isna(wrong_letter):
            continue
        wrong_letter = str(wrong_letter).strip().upper()
        if len(wrong_letter) != 1:
            continue
        for layer_idx, ds_wrong in compute_wrong_ds_per_layer(layer_logits, wrong_letter):
            rows.append({"layer": layer_idx, "ds_chosen_wrong": ds_wrong})
    return rows


def build_default_authority_out_plot(dataset: str, data_seed: int) -> str:
    return f"figure/ds_authority_{dataset}_rtn_w4_{data_seed}.png"


def collect_authority_ds_curves(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], str] | None:
    root = Path(args.output_inference_root).resolve()
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    for level in LEVELS:
        pkl_path = (
            root
            / args.dataset
            / "prefix_and_opinion"
            / "academic"
            / "original"
            / level
            / f"{args.model_output_name}_logit_all_{args.data_seed}.pkl"
        )
        if not pkl_path.exists():
            print(f"file missing, skipping {level}: {pkl_path}")
            continue

        rows = load_chosen_wrong_rows(pkl_path)
        if not rows:
            print(f"no usable data, skipping {level}: {pkl_path}")
            continue

        by = pd.DataFrame(rows).groupby("layer").agg({"ds_chosen_wrong": "mean"}).reset_index().sort_values("layer")
        layers = by["layer"].to_numpy()
        ds_vals = by["ds_chosen_wrong"].to_numpy()
        curves.append((level, layers, ds_vals))

    if not curves:
        print("no curves to plot.")
        return None

    out_plot = args.out_plot.strip() or build_default_authority_out_plot(args.dataset, args.data_seed)
    return curves, out_plot


def build_authority_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="authority three-level chosen_wrong DS")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--model_output_name", type=str, default="llama_3.1_8b_instruct_rtn_w4")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--out_plot", type=str, default="")
    return parser
