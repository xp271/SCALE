"""FP vs quant: four DS curves (correct / chosen wrong). Plotting: ``fp_quant_ds_render``."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from compute_decision_score import decision_score
from mcq_option_utils import contiguous_letters_from_logits, is_single_upper_option_letter

COLOR_FP = "#0072B2"
COLOR_QUANT = "#D55E00"


def _compute_letter_ds_per_layer(layer_logits: dict, letter: str):
    if not isinstance(layer_logits, dict):
        return []
    letter = str(letter).strip().upper()
    if len(letter) != 1:
        return []
    out = []
    for key, logits in layer_logits.items():
        if not isinstance(logits, dict) or contiguous_letters_from_logits(logits) is None:
            continue
        if not is_single_upper_option_letter(letter, logits):
            continue
        try:
            layer_idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        ds = decision_score(logits.get(letter, 0.0), logits)
        out.append((layer_idx, ds))
    return sorted(out, key=lambda x: x[0])


def _load_ds_for_letter_column(pkl_path: str, letter_column: str):
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"read failed: {pkl_path} ({e})")
        return []
    if "layer_logits" not in df.columns or letter_column not in df.columns:
        return []

    rows = []
    for _, row in df.iterrows():
        layer_logits = row.get("layer_logits")
        letter = row.get(letter_column)
        if pd.isna(layer_logits) or pd.isna(letter):
            continue
        for layer_idx, ds_val in _compute_letter_ds_per_layer(layer_logits, letter):
            rows.append({"layer": layer_idx, "ds": ds_val})
    return rows


def _question_align_key(df: pd.DataFrame) -> pd.Series:
    for col in ("question_id", "id", "question", "query", "full_question"):
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series([f"__idx__{i}" for i in range(len(df))], index=df.index, dtype="object")


def _chosen_wrong_lookup_from_opinion(opinion_df: pd.DataFrame) -> pd.Series:
    if "chosen_wrong_answer_index" not in opinion_df.columns:
        return pd.Series(dtype=object)
    keys = _question_align_key(opinion_df)
    wrong = opinion_df["chosen_wrong_answer_index"].astype(str).str.strip().str.upper()
    tbl = pd.DataFrame({"k": keys.values, "w": wrong.values})
    tbl = tbl.dropna(subset=["w"])
    tbl = tbl[tbl["w"].str.len() == 1]
    tbl = tbl[tbl["w"].str.match(r"[A-Z]", na=False)]
    tbl = tbl.drop_duplicates(subset=["k"], keep="first")
    return tbl.set_index("k")["w"]


def _align_chosen_wrong_onto_plain(plain_df: pd.DataFrame, opinion_df: pd.DataFrame) -> pd.Series:
    lookup = _chosen_wrong_lookup_from_opinion(opinion_df)
    keys = _question_align_key(plain_df)
    out = keys.map(lookup)
    out.index = plain_df.index
    return out


def _plain_chosen_wrong_letter_series(plain_df: pd.DataFrame, opinion_df: pd.DataFrame) -> pd.Series:
    if "chosen_wrong_answer_index" in plain_df.columns and plain_df["chosen_wrong_answer_index"].notna().any():
        return plain_df["chosen_wrong_answer_index"]
    return _align_chosen_wrong_onto_plain(plain_df, opinion_df)


def _load_plain_ds_chosen_wrong_aligned(plain_pkl: str, opinion_pkl: str) -> list[dict]:
    try:
        df_p = pd.read_pickle(plain_pkl)
        df_o = pd.read_pickle(opinion_pkl)
    except Exception as e:
        print(f"read failed: plain={plain_pkl} opinion={opinion_pkl} ({e})")
        return []
    if "layer_logits" not in df_p.columns:
        return []

    used_align = not (
        "chosen_wrong_answer_index" in df_p.columns and df_p["chosen_wrong_answer_index"].notna().any()
    )
    wrong_series = _plain_chosen_wrong_letter_series(df_p, df_o)
    if used_align:
        n_ok = wrong_series.notna().sum()
        print(
            f"[chosen_wrong·plain] wrong option letters from opinion: {plain_pkl} ← {opinion_pkl} "
            f"({int(n_ok)}/{len(df_p)} rows mapped)"
        )

    rows: list[dict] = []
    for idx, row in df_p.iterrows():
        layer_logits = row.get("layer_logits")
        raw_w = wrong_series.loc[idx]
        if pd.isna(layer_logits) or pd.isna(raw_w):
            continue
        letter = str(raw_w).strip().upper()
        if len(letter) != 1 or not letter.isalpha():
            continue
        for layer_idx, ds_val in _compute_letter_ds_per_layer(layer_logits, letter):
            rows.append({"layer": layer_idx, "ds": ds_val})
    return rows


def _mean_ds_by_layer(rows: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    if not rows:
        return None
    by = pd.DataFrame(rows).groupby("layer").agg({"ds": "mean"}).reset_index().sort_values("layer")
    return by["layer"].to_numpy(), by["ds"].to_numpy()


def _normalize_bit(bit: str) -> str:
    b = (bit or "").strip().lower()
    if re.fullmatch(r"\d+", b):
        return f"w{b}"
    return b


def _expected_stems(model_id: str, method: str, bit: str, seed: int) -> tuple[str, str]:
    method = method.strip().lower()
    return (
        f"{model_id}_full_precision_logit_all_{seed}",
        f"{model_id}_{method}_{bit}_logit_all_{seed}",
    )


def _cond_legend_slug(condition_slug: str) -> str:
    s = (condition_slug or "").strip()
    if s == "opinion_only":
        return "opinion"
    return s


def _legend_paren(inner: str) -> str:
    return f" ({inner})"


def _legend_label_fp(condition_slug: str) -> str:
    return "FP" + _legend_paren(_cond_legend_slug(condition_slug))


def _legend_label_quant(method: str, _bit: str, condition_slug: str) -> str:
    m = method.strip().lower().upper()
    return m + _legend_paren(_cond_legend_slug(condition_slug))


def _find_pkl(condition_dir: Path, stem: str) -> Path | None:
    if not condition_dir.is_dir():
        return None
    direct = condition_dir / f"{stem}.pkl"
    if direct.is_file():
        return direct
    matches = sorted(condition_dir.rglob(f"{stem}.pkl"))
    return matches[0] if matches else None


def collect_fp_quant_ds_series(
    args: argparse.Namespace,
) -> tuple[list[tuple[np.ndarray, np.ndarray, str, str, str]], list[tuple[np.ndarray, np.ndarray, str, str, str]], Path, Path] | None:
    root = Path(args.output_inference_root).resolve()
    dataset = args.dataset.strip()
    model_id = args.model_id.strip()
    bit = _normalize_bit(args.bit)
    seed = args.data_seed
    fp_stem, q_stem = _expected_stems(model_id, args.method, bit, seed)

    plain_dir = root / dataset / "plain"
    opinion_dir = root / dataset / "opinion_only"

    paths = {
        "fp_plain": _find_pkl(plain_dir, fp_stem),
        "fp_opinion": _find_pkl(opinion_dir, fp_stem),
        "q_plain": _find_pkl(plain_dir, q_stem),
        "q_opinion": _find_pkl(opinion_dir, q_stem),
    }
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        print("missing pkls (check dirs and filenames):")
        for k in missing:
            sub = opinion_dir if k.endswith("_opinion") else plain_dir
            stem = fp_stem if k.startswith("fp") else q_stem
            print(f"  [{k}] expected {sub}/{stem}.pkl")
        return None

    method_u = args.method.strip().lower()
    cond_plain = "plain"
    cond_opinion = "opinion_only"

    specs_c = [
        (_legend_label_fp(cond_plain), paths["fp_plain"], COLOR_FP, "-"),
        (_legend_label_fp(cond_opinion), paths["fp_opinion"], COLOR_FP, "--"),
        (_legend_label_quant(method_u, bit, cond_plain), paths["q_plain"], COLOR_QUANT, "-"),
        (_legend_label_quant(method_u, bit, cond_opinion), paths["q_opinion"], COLOR_QUANT, "--"),
    ]
    series_correct: list[tuple[np.ndarray, np.ndarray, str, str, str]] = []
    for legend, pkl_path, color, ls in specs_c:
        rows = _load_ds_for_letter_column(str(pkl_path), "correct_answer_index")
        curve = _mean_ds_by_layer(rows)
        if curve is None:
            print(f"no usable layer_logits: {pkl_path} (correct_answer_index)")
            return None
        layers, ds_vals = curve
        series_correct.append((layers, ds_vals, legend, color, ls))

    items_w = [
        (_legend_label_fp(cond_plain), paths["fp_plain"], paths["fp_opinion"], COLOR_FP, "-", True),
        (_legend_label_fp(cond_opinion), paths["fp_opinion"], None, COLOR_FP, "--", False),
        (_legend_label_quant(method_u, bit, cond_plain), paths["q_plain"], paths["q_opinion"], COLOR_QUANT, "-", True),
        (_legend_label_quant(method_u, bit, cond_opinion), paths["q_opinion"], None, COLOR_QUANT, "--", False),
    ]
    series_wrong: list[tuple[np.ndarray, np.ndarray, str, str, str]] = []
    for legend, p_plain_or_main, p_op_partner, color, ls, is_plain_branch in items_w:
        if is_plain_branch:
            rows = _load_plain_ds_chosen_wrong_aligned(str(p_plain_or_main), str(p_op_partner))
            tag = f"{p_plain_or_main} (chosen_wrong aligned to {p_op_partner})"
        else:
            rows = _load_ds_for_letter_column(str(p_plain_or_main), "chosen_wrong_answer_index")
            tag = str(p_plain_or_main)
        curve = _mean_ds_by_layer(rows)
        if curve is None:
            print(f"no usable chosen_wrong DS data: {tag}")
            return None
        layers, ds_vals = curve
        series_wrong.append((layers, ds_vals, legend, color, ls))

    figure_dir = Path(args.figure_dir)
    mid = f"{dataset}_{model_id}_{method_u}_{bit}_{seed}"
    out_c = Path(args.out_correct) if args.out_correct else figure_dir / f"ds_correct_answer_{mid}.png"
    out_w = Path(args.out_chosen_wrong) if args.out_chosen_wrong else figure_dir / f"ds_chosen_wrong_{mid}.png"

    return series_correct, series_wrong, out_c, out_w


def build_fp_quant_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FP vs specified quant: plain/opinion DS (correct option & chosen wrong)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--bit", type=str, required=True)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--figure_dir", type=str, default="figure")
    parser.add_argument("--out_correct", type=str, default="")
    parser.add_argument("--out_chosen_wrong", type=str, default="")
    return parser
