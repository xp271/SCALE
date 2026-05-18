"""
Compare chosen_wrong DS by bit/method (no matplotlib).
plotting：`figure/mechanistic/extras/ds_across_methods_render`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from compute_decision_score import decision_score
from mcq_option_utils import contiguous_letters_from_logits, is_single_upper_option_letter

EPS = 1e-9
METHOD_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
BASELINE_COLOR = "#222222"
VALID_BITS = ("w4", "w6", "w8")


def _compute_wrong_ds_per_layer(layer_logits: dict, wrong_letter: str):
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


def _load_method_wrong_ds(pkl_path: str):
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"read failed, skipping: {pkl_path} ({e})")
        return []
    if "layer_logits" not in df.columns or "chosen_wrong_answer_index" not in df.columns:
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
        for layer_idx, ds_wrong in _compute_wrong_ds_per_layer(layer_logits, wrong_letter):
            rows.append({"layer": layer_idx, "ds_chosen_wrong": ds_wrong})
    return rows


def _collect_candidates_by_method(opinion_dir: Path, model_id: str, bit: str, data_seed: int):
    seed_suffix = f"_{data_seed}"
    quantized = {}
    full_precision = None
    pattern = re.compile(rf"^{re.escape(model_id)}_(.+)_{re.escape(bit)}_logit_all_{data_seed}$")
    fp_stem = f"{model_id}_full_precision_logit_all_{data_seed}"

    for p in opinion_dir.rglob("*.pkl"):
        stem = p.stem
        if not stem.endswith(seed_suffix):
            continue
        if stem == fp_stem:
            full_precision = str(p)
            continue
        m = pattern.match(stem)
        if not m:
            continue
        method = m.group(1)
        quantized[method] = str(p)

    return quantized, full_precision


def _collect_candidates_by_bit(opinion_dir: Path, model_id: str, method: str, data_seed: int):
    seed_suffix = f"_{data_seed}"
    quantized = {}
    full_precision = None
    pattern = re.compile(rf"^{re.escape(model_id)}_{re.escape(method)}_(w[0-9]+)_logit_all_{data_seed}$")
    fp_stem = f"{model_id}_full_precision_logit_all_{data_seed}"

    for p in opinion_dir.rglob("*.pkl"):
        stem = p.stem
        if not stem.endswith(seed_suffix):
            continue
        if stem == fp_stem:
            full_precision = str(p)
            continue
        m = pattern.match(stem)
        if not m:
            continue
        bit = m.group(1)
        quantized[bit] = str(p)

    return quantized, full_precision


def _path_with_seed(path: str, seed: int):
    if not path:
        return path
    p = Path(path)
    suffix = f"_{seed}"
    if p.stem.endswith(suffix):
        return str(p)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def _build_default_output_paths(
    dataset: str,
    model_id: str,
    compare_mode: str,
    bit: str,
    method: str,
    data_seed: int,
):
    out_dir = Path("figure")
    if compare_mode == "by_method":
        base = f"ds_chosen_wrong_{dataset}_{model_id}_by_method_{bit}_{data_seed}"
    else:
        base = f"ds_chosen_wrong_{dataset}_{model_id}_by_bit_{method}_{data_seed}"
    return str(out_dir / f"{base}.png")


def collect_decision_score_across_methods_curves(args: argparse.Namespace) -> tuple[list[tuple[np.ndarray, np.ndarray, str, str]], str, str] | None:
    """
    Returns ``(curves, out_plot, compare_mode)`` where each curve is ``(layers, ds_vals, raw_label, matplotlib_color_hex)``.
    """
    opinion_dir = Path(args.output_inference_root).resolve() / args.dataset / "opinion_only"
    if not opinion_dir.exists():
        print(f"directory does not exist: {opinion_dir}")
        return None

    bit = ""
    method = ""

    if args.compare_mode == "by_method":
        bit = args.bit.strip().lower()
        if not bit:
            print("--bit required in by_method mode.")
            return None
        if bit not in VALID_BITS:
            print(f"--bit should be one of {VALID_BITS}, got: {bit}")
            return None
        quantized_paths, fp_path = _collect_candidates_by_method(opinion_dir, args.model_id, bit, args.data_seed)
    else:
        method = args.method.strip().lower()
        if not method:
            print("--method required in by_bit mode.")
            return None
        quantized_paths, fp_path = _collect_candidates_by_bit(opinion_dir, args.model_id, method, args.data_seed)
        quantized_paths = dict(sorted(quantized_paths.items(), key=lambda kv: (kv[0] not in VALID_BITS, kv[0])))

    if not quantized_paths:
        if args.compare_mode == "by_method":
            print(f"no matching files: model_id={args.model_id}, bit={bit}, seed={args.data_seed}. Check {opinion_dir}.")
        else:
            print(f"no matching files: model_id={args.model_id}, method={method}, seed={args.data_seed}. Check {opinion_dir}.")
        return None

    default_out_plot = _build_default_output_paths(
        dataset=args.dataset,
        model_id=args.model_id,
        compare_mode=args.compare_mode,
        bit=bit if args.compare_mode == "by_method" else "",
        method=method if args.compare_mode == "by_bit" else "",
        data_seed=args.data_seed,
    )

    curves: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for i, key_name in enumerate(sorted(quantized_paths.keys())):
        pkl_path = quantized_paths[key_name]
        rows = _load_method_wrong_ds(pkl_path)
        if not rows:
            print(f"skip {key_name}: no valid layer_logits or chosen_wrong_answer_index.")
            continue
        by = (
            pd.DataFrame(rows).groupby("layer").agg({"ds_chosen_wrong": "mean"}).reset_index().sort_values("layer")
        )
        layers = by["layer"].to_numpy()
        ds_vals = by["ds_chosen_wrong"].to_numpy()
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        curves.append((layers, ds_vals, key_name, color))

    if args.include_full_precision and fp_path:
        fp_rows = _load_method_wrong_ds(fp_path)
        if fp_rows:
            by_fp = (
                pd.DataFrame(fp_rows).groupby("layer").agg({"ds_chosen_wrong": "mean"}).reset_index().sort_values("layer")
            )
            layers_fp = by_fp["layer"].to_numpy()
            ds_fp = by_fp["ds_chosen_wrong"].to_numpy()
            curves.append((layers_fp, ds_fp, "full_precision", BASELINE_COLOR))
        else:
            print("warning: full precision file exists but no usable data; skipped baseline.")
    elif args.include_full_precision and not fp_path:
        print("warning: full precision file not found; skipped baseline.")

    if not curves:
        print("no curves to plot.")
        return None

    out_plot = _path_with_seed(args.out_plot, args.data_seed) if args.out_plot else default_out_plot
    return (curves, out_plot, args.compare_mode)


def build_ds_across_methods_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare chosen_wrong_answer Decision Score (same bit / different methods or vice versa)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--compare_mode", type=str, default="by_method", choices=["by_method", "by_bit"])
    parser.add_argument("--bit", type=str, default="", help="required for by_method")
    parser.add_argument("--method", type=str, default="", help="required for by_bit")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--include_full_precision", action="store_true")
    parser.add_argument("--out_plot", type=str, default="", help="Output figure path")
    return parser
