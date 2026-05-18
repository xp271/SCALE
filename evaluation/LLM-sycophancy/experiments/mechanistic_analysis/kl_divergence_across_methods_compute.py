"""Compare KL(Opinion||Plain) aligned by method/bit (compute-only). Plotting: ``kl_across_methods_render``."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from kl_divergence_compute import compute_kl_per_layer


def _safe_read_pickle(path_str: str):
    try:
        return pd.read_pickle(path_str)
    except Exception as e:
        print(f"read failed, skipping: {path_str} ({e})")
        return None


def _collect_seed_stem_map(directory: Path, seed: int):
    out = {}
    suffix = f"_{seed}"
    for p in directory.rglob("*.pkl"):
        stem = p.stem
        if stem.endswith(suffix):
            out[stem] = str(p)
    return out


def _collect_pairs_by_method(plain_map: dict, opinion_map: dict, model_id: str, bit: str, seed: int):
    pairs = {}
    pattern = re.compile(rf"^{re.escape(model_id)}_(.+)_{re.escape(bit)}_logit_all_{seed}$")
    fp_stem = f"{model_id}_full_precision_logit_all_{seed}"
    fp_pair = None

    if fp_stem in plain_map and fp_stem in opinion_map:
        fp_pair = (plain_map[fp_stem], opinion_map[fp_stem])

    for stem, opinion_path in opinion_map.items():
        m = pattern.match(stem)
        if not m:
            continue
        if stem not in plain_map:
            continue
        method = m.group(1)
        pairs[method] = (plain_map[stem], opinion_path)
    return pairs, fp_pair


def _collect_pairs_by_bit(plain_map: dict, opinion_map: dict, model_id: str, method: str, seed: int):
    pairs = {}
    pattern = re.compile(rf"^{re.escape(model_id)}_{re.escape(method)}_(w[0-9]+)_logit_all_{seed}$")
    fp_stem = f"{model_id}_full_precision_logit_all_{seed}"
    fp_pair = None

    if fp_stem in plain_map and fp_stem in opinion_map:
        fp_pair = (plain_map[fp_stem], opinion_map[fp_stem])

    for stem, opinion_path in opinion_map.items():
        m = pattern.match(stem)
        if not m:
            continue
        if stem not in plain_map:
            continue
        bit = m.group(1)
        pairs[bit] = (plain_map[stem], opinion_path)
    return pairs, fp_pair


def _default_out_plot(dataset: str, model_id: str, compare_mode: str, bit: str, method: str, seed: int):
    out_dir = Path("figure")
    if compare_mode == "by_method":
        name = f"kl_divergence_{dataset}_{model_id}_by_method_{bit}_{seed}.png"
    else:
        name = f"kl_divergence_{dataset}_{model_id}_by_bit_{method}_{seed}.png"
    return str(out_dir / name)


VALID_BITS = ("w4", "w6", "w8")
CURVE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
FP_COLOR = "#222222"


def collect_kl_divergence_across_methods_curves(
    args: argparse.Namespace,
) -> tuple[list[tuple[list, list, str, str]], str] | None:
    plain_dir = Path(args.output_inference_root).resolve() / args.dataset / "plain"
    opinion_dir = Path(args.output_inference_root).resolve() / args.dataset / "opinion_only"
    if not plain_dir.exists() or not opinion_dir.exists():
        print(f"directory missing: plain={plain_dir}, opinion={opinion_dir}")
        return None

    plain_map = _collect_seed_stem_map(plain_dir, args.data_seed)
    opinion_map = _collect_seed_stem_map(opinion_dir, args.data_seed)
    if not plain_map or not opinion_map:
        print(f"no pkls for seed={args.data_seed}: plain={len(plain_map)}, opinion={len(opinion_map)}")
        return None

    if args.compare_mode == "by_method":
        bit = args.bit.strip().lower()
        if not bit:
            print("--bit required in by_method mode.")
            return None
        if bit not in VALID_BITS:
            print(f"--bit should be one of {VALID_BITS}, got: {bit}")
            return None
        pairs, fp_pair = _collect_pairs_by_method(plain_map, opinion_map, args.model_id, bit, args.data_seed)
        sorted_keys = sorted(pairs.keys())
        method = ""
    else:
        method = args.method.strip().lower()
        if not method:
            print("--method required in by_bit mode.")
            return None
        pairs, fp_pair = _collect_pairs_by_bit(plain_map, opinion_map, args.model_id, method, args.data_seed)
        sorted_keys = sorted(pairs.keys(), key=lambda x: (x not in VALID_BITS, x))
        bit = ""

    if not pairs:
        if args.compare_mode == "by_method":
            print(f"no matching files: model_id={args.model_id}, bit={bit}, seed={args.data_seed}")
        else:
            print(f"no matching files: model_id={args.model_id}, method={method}, seed={args.data_seed}")
        return None

    curves = []
    for i, key in enumerate(sorted_keys):
        plain_path, opinion_path = pairs[key]
        df_plain = _safe_read_pickle(plain_path)
        df_opinion = _safe_read_pickle(opinion_path)
        if df_plain is None or df_opinion is None:
            continue
        layers, mean_kl = compute_kl_per_layer(df_plain, df_opinion, max_rows=args.max_rows)
        if not layers:
            print(f"skip {key}: no valid layer_logits.")
            continue
        curves.append((layers, mean_kl, key, CURVE_COLORS[i % len(CURVE_COLORS)]))

    if not args.no_full_precision:
        if fp_pair is None:
            print("warning: no full-precision plain/opinion pair; skipped FP baseline.")
        else:
            df_fp_plain = _safe_read_pickle(fp_pair[0])
            df_fp_opinion = _safe_read_pickle(fp_pair[1])
            if df_fp_plain is not None and df_fp_opinion is not None:
                layers_fp, mean_kl_fp = compute_kl_per_layer(df_fp_plain, df_fp_opinion, max_rows=args.max_rows)
                if layers_fp:
                    curves.append((layers_fp, mean_kl_fp, "full_precision", FP_COLOR))
                else:
                    print("warning: full precision files exist but no usable layer_logits; skipped FP baseline.")

    if not curves:
        print("no curves to plot.")
        return None

    out_plot = args.out_plot if args.out_plot else _default_out_plot(
        args.dataset, args.model_id, args.compare_mode, bit, method, args.data_seed
    )
    return curves, out_plot


def build_kl_across_methods_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare KL(Opinion || Plain): same bit different methods / same method different bits")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--compare_mode", type=str, default="by_method", choices=["by_method", "by_bit"])
    parser.add_argument("--bit", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--no_full_precision", action="store_true")
    parser.add_argument("--out_plot", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=None)
    return parser
