"""
用 run_syco_logit_cot.py 输出的 pkl（含 layer_logits）计算 Decision Score (DS)。

公式: DS(x) = (l_x - min(l_A,l_B,l_C,l_D)) / (max(...) - min(...) + ε), ε=1e-9
支持 opinion_only（正确答案 + 用户声称答案）和 plain（仅正确答案）；可把 plain 与 opinion 合并画在一张图里。

用法（在 experiments/mechanistic_analysis 下或 cwd 在项目根以便 import mcq_option_utils）:
  python compute_decision_score.py output_inference/mmlu/opinion_only/*.pkl --out_plot ds.png
  python compute_decision_score.py --plain .../plain/*.pkl --opinion .../opinion_only/*.pkl --out_plot ds_merged.png
需画 KL 时请用仓库根目录: ``python -m figure.mechanistic.kl_plot ...``。
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from collections.abc import Iterable

from mcq_option_utils import contiguous_letters_from_logits, is_single_upper_option_letter

EPS = 1e-9

COLOR_CORRECT_PLAIN = "#d62728"
COLOR_WRONG_PLAIN = "#ff9896"
COLOR_CORRECT_OPINION = "#1f77b4"
COLOR_WRONG_OPINION = "#17becf"


def decision_score(l_x: float, l_all: dict) -> float:
    """DS(x) = (l_x - min) / (max - min + ε). l_all: dict like {'A': ..., 'B': ..., 'C': ..., 'D': ...}."""
    vals = list(l_all.values())
    mn, mx = min(vals), max(vals)
    if mx - mn + EPS == 0:
        return 0.0
    return (l_x - mn) / (mx - mn + EPS)


def compute_ds_per_layer_opinion(layer_logits: dict, correct_letter: str, syco_letter: str):
    """Returns list of (layer_idx, ds_correct, ds_sycophantic)."""
    if not isinstance(layer_logits, dict):
        return []
    out = []
    for key, logits in layer_logits.items():
        if not isinstance(logits, dict) or contiguous_letters_from_logits(logits) is None:
            continue
        if not is_single_upper_option_letter(correct_letter, logits) or not is_single_upper_option_letter(
            syco_letter, logits
        ):
            continue
        try:
            layer_idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        ds_correct = decision_score(logits.get(correct_letter, 0.0), logits)
        ds_sycophantic = decision_score(logits.get(syco_letter, 0.0), logits)
        out.append((layer_idx, ds_correct, ds_sycophantic))
    return sorted(out, key=lambda x: x[0])


def compute_ds_per_layer_plain(layer_logits: dict, correct_letter: str):
    """Plain 只有正确答案，返回 list of (layer_idx, ds_correct)."""
    if not isinstance(layer_logits, dict):
        return []
    out = []
    for key, logits in layer_logits.items():
        if not isinstance(logits, dict) or contiguous_letters_from_logits(logits) is None:
            continue
        if not is_single_upper_option_letter(correct_letter, logits):
            continue
        try:
            layer_idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        ds_correct = decision_score(logits.get(correct_letter, 0.0), logits)
        out.append((layer_idx, ds_correct))
    return sorted(out, key=lambda x: x[0])


def _load_plain_pkls(path_strs, wrong_letters_per_row=None):
    """
    返回 list of dict with layer, ds_correct_plain, 以及（若提供 wrong_letters_per_row）ds_wrong_plain。
    wrong_letters_per_row: list 与 plain 行一一对应，表示每题在 opinion 里的 chosen wrong 选项，用于画 Chosen Wrong (Plain)。
    """
    correct_col = "correct_answer_index"
    rows = []
    for path_str in path_strs:
        p = Path(path_str)
        if not p.exists():
            continue
        df = pd.read_pickle(p)
        if "layer_logits" not in df.columns or correct_col not in df.columns:
            continue
        for idx, row in df.iterrows():
            layer_logits, correct_letter = row.get("layer_logits"), row.get(correct_col)
            if pd.isna(layer_logits) or pd.isna(correct_letter):
                continue
            correct_letter = str(correct_letter).strip().upper()
            wrong_letter = None
            if wrong_letters_per_row is not None and int(idx) < len(wrong_letters_per_row):
                w = str(wrong_letters_per_row[int(idx)]).strip().upper()
                if len(w) == 1 and w.isupper() and w.isalpha():
                    wrong_letter = w
            for key, logits in layer_logits.items():
                if not isinstance(logits, dict) or contiguous_letters_from_logits(logits) is None:
                    continue
                if not is_single_upper_option_letter(correct_letter, logits):
                    continue
                try:
                    layer_idx = int(key.split("_")[1])
                except (IndexError, ValueError):
                    continue
                ds_c = decision_score(logits.get(correct_letter, 0.0), logits)
                r = {"layer": layer_idx, "ds_correct_plain": ds_c}
                r["ds_wrong_plain"] = (
                    decision_score(logits.get(wrong_letter, 0.0), logits)
                    if wrong_letter and is_single_upper_option_letter(wrong_letter, logits)
                    else None
                )
                rows.append(r)
    return rows


def _load_opinion_pkls(path_strs):
    """返回 list of dict with layer, ds_correct, ds_sycophantic."""
    correct_col = "correct_answer_index"
    syco_col = "chosen_wrong_answer_index"
    rows = []
    for path_str in path_strs:
        p = Path(path_str)
        if not p.exists():
            continue
        df = pd.read_pickle(p)
        if "layer_logits" not in df.columns or correct_col not in df.columns or syco_col not in df.columns:
            continue
        for _, row in df.iterrows():
            layer_logits = row.get("layer_logits")
            correct_letter = row.get(correct_col)
            syco_letter = row.get(syco_col)
            if pd.isna(layer_logits) or pd.isna(correct_letter) or pd.isna(syco_letter):
                continue
            correct_letter = str(correct_letter).strip().upper()
            syco_letter = str(syco_letter).strip().upper()
            if len(correct_letter) != 1 or len(syco_letter) != 1:
                continue
            for layer_idx, ds_c, ds_s in compute_ds_per_layer_opinion(layer_logits, correct_letter, syco_letter):
                rows.append({"layer": layer_idx, "ds_correct": ds_c, "ds_sycophantic": ds_s})
    return rows


def compute_and_plot_decision_score(
    *,
    plain_paths: Iterable[str] | None = None,
    opinion_paths: Iterable[str] | None = None,
    out_csv: str = "",
    out_plot: str = "",
    data_seed: int | None = None,
) -> None:
    """加载 pkl、打印每层均值、可选写 CSV/plot。"""
    plain_paths = list(plain_paths or [])
    opinion_paths = list(opinion_paths or [])
    has_plain = len(plain_paths) > 0
    has_opinion = len(opinion_paths) > 0

    if not has_opinion and not has_plain:
        print("请提供至少 plain 路径或 opinion 路径。")
        return

    # 加载 opinion
    opinion_rows = _load_opinion_pkls(opinion_paths) if has_opinion else []
    # 若同时有 plain 和 opinion，用 opinion 的 chosen_wrong 与 plain 按行对齐，得到 Chosen Wrong (Plain)
    wrong_per_row = None
    if has_plain and has_opinion and opinion_paths:
        first_opinion = Path(opinion_paths[0])
        if first_opinion.exists():
            try:
                odf = pd.read_pickle(first_opinion)
                if "chosen_wrong_answer_index" in odf.columns:
                    wrong_per_row = odf["chosen_wrong_answer_index"].astype(str).str.strip().str.upper().tolist()
            except Exception:
                pass
    plain_rows = _load_plain_pkls(plain_paths, wrong_letters_per_row=wrong_per_row) if has_plain else []

    if has_opinion and opinion_rows:
        tbl_o = pd.DataFrame(opinion_rows)
        by_o = tbl_o.groupby("layer").agg({"ds_correct": "mean", "ds_sycophantic": "mean"}).reset_index().sort_values("layer")
        print("Decision Score 每层均值 (Opinion):")
        print(by_o.to_string(index=False))
    if has_plain and plain_rows:
        tbl_p = pd.DataFrame(plain_rows)
        by_p = tbl_p.groupby("layer").agg({"ds_correct_plain": "mean"}).reset_index().sort_values("layer")
        if "ds_wrong_plain" in tbl_p.columns and tbl_p["ds_wrong_plain"].notna().any():
            by_p = tbl_p.groupby("layer").agg({"ds_correct_plain": "mean", "ds_wrong_plain": "mean"}).reset_index().sort_values("layer")
            print("Decision Score 每层均值 (Plain): Correct + Chosen Wrong (对齐 opinion 的错选项):")
        else:
            print("Decision Score 每层均值 (Plain, 仅正确):")
        print(by_p.to_string(index=False))

    def _path_with_seed(path: str, seed: int | None) -> str:
        if not path or seed is None:
            return path
        p = Path(path)
        return str(p.parent / f"{p.stem}_{seed}{p.suffix}")

    resolved_out_csv = _path_with_seed(out_csv, data_seed)
    resolved_out_plot = _path_with_seed(out_plot, data_seed)

    if resolved_out_csv and has_opinion and opinion_rows:
        pd.DataFrame(opinion_rows).groupby("layer").agg({"ds_correct": "mean", "ds_sycophantic": "mean"}).reset_index().sort_values(
            "layer"
        ).to_csv(resolved_out_csv, index=False)
        print(f"已保存: {resolved_out_csv}")

    if not resolved_out_plot:
        return

    from _bootstrap_repo import ensure_project_syspath

    ensure_project_syspath(origin_file=Path(__file__))
    from figure.mechanistic.render_decision_score import save_decision_score_figure

    if has_plain and plain_rows and has_opinion and opinion_rows:
        agg_p = {"ds_correct_plain": "mean"}
        df_plain = pd.DataFrame(plain_rows)
        if "ds_wrong_plain" in df_plain.columns and df_plain["ds_wrong_plain"].notna().any():
            agg_p["ds_wrong_plain"] = "mean"
        by_p = df_plain.groupby("layer").agg(agg_p).reset_index().sort_values("layer")
        by_o = pd.DataFrame(opinion_rows).groupby("layer").agg({"ds_correct": "mean", "ds_sycophantic": "mean"}).reset_index().sort_values("layer")
        layers_plain = by_p["layer"].values
        layers_o = by_o["layer"].values
        series = [
            {"layers": layers_plain, "values": by_p["ds_correct_plain"].values, "color": COLOR_CORRECT_PLAIN, "linewidth": 2, "markersize": 4, "label": "Correct Answer (Plain)"},
        ]
        if "ds_wrong_plain" in by_p.columns:
            series.append({"layers": layers_plain, "values": by_p["ds_wrong_plain"].values, "color": COLOR_WRONG_PLAIN, "marker": "x", "linestyle": "--", "linewidth": 1.5, "markersize": 4, "label": "Chosen Wrong Answer (Plain)"})
        series.extend(
            [
                {"layers": layers_o, "values": by_o["ds_correct"].values, "color": COLOR_CORRECT_OPINION, "linewidth": 2, "markersize": 4, "label": "Correct Answer (Opinion)"},
                {"layers": layers_o, "values": by_o["ds_sycophantic"].values, "color": COLOR_WRONG_OPINION, "marker": "x", "linestyle": "--", "linewidth": 1.5, "markersize": 4, "label": "Chosen Wrong Answer (Opinion)"},
            ]
        )
        save_decision_score_figure(
            out_plot=resolved_out_plot,
            title="Score of Correct vs Chosen Wrong Answer by Layer (Opinion vs Plain)",
            series=series,
            xlim=(float(min(layers_plain.min(), layers_o.min())), float(max(layers_plain.max(), layers_o.max()))),
        )
    elif has_opinion and opinion_rows:
        by_o = pd.DataFrame(opinion_rows).groupby("layer").agg({"ds_correct": "mean", "ds_sycophantic": "mean"}).reset_index().sort_values("layer")
        layers = by_o["layer"].values
        series = [
            {"layers": layers, "values": by_o["ds_correct"].values, "color": COLOR_CORRECT_PLAIN, "linewidth": 2, "markersize": 5, "label": "Correct Answer (Opinion)"},
            {"layers": layers, "values": by_o["ds_sycophantic"].values, "color": COLOR_CORRECT_OPINION, "marker": "x", "linestyle": "--", "linewidth": 1.5, "markersize": 5, "label": "Chosen Wrong Answer (Opinion)"},
        ]
        save_decision_score_figure(
            out_plot=resolved_out_plot,
            title="Score of Correct vs Chosen Wrong Answer by Layer (Opinion)",
            series=series,
            xlim=(float(layers.min()), float(layers.max())),
        )
    elif has_plain and plain_rows:
        by_p = pd.DataFrame(plain_rows).groupby("layer").agg({"ds_correct_plain": "mean"}).reset_index().sort_values("layer")
        layers = by_p["layer"].values
        series = [{"layers": layers, "values": by_p["ds_correct_plain"].values, "color": COLOR_CORRECT_PLAIN, "linewidth": 2, "markersize": 5, "label": "Correct Answer (Plain)"}]
        save_decision_score_figure(
            out_plot=resolved_out_plot,
            title="Decision Score by Layer (Plain)",
            series=series,
            xlim=(float(layers.min()), float(layers.max())),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="从 run_syco_logit_cot 输出的 pkl 计算 Decision Score，可合并 plain 与 opinion")
    parser.add_argument("pkl_paths", type=str, nargs="*", help="opinion_only 的 .pkl（与 --opinion 二选一或同时用）")
    parser.add_argument("--plain", type=str, nargs="+", default=None, help="plain 的 .pkl，可与 --opinion 合并画一张图")
    parser.add_argument("--opinion", type=str, nargs="+", default=None, help="opinion_only 的 .pkl")
    parser.add_argument("--out_csv", type=str, default="", help="可选：每层均值保存为 CSV")
    parser.add_argument("--out_plot", type=str, default="", help="可选：图保存路径")
    parser.add_argument("--data_seed", type=int, default=None, help="数据种子：输出 CSV/图文件名会带 _${seed} 后缀")
    args = parser.parse_args()

    opinion_paths = args.opinion if args.opinion is not None else args.pkl_paths
    plain_paths_list = args.plain or []

    compute_and_plot_decision_score(
        plain_paths=plain_paths_list,
        opinion_paths=list(opinion_paths),
        out_csv=args.out_csv,
        out_plot=args.out_plot,
        data_seed=args.data_seed,
    )


if __name__ == "__main__":
    main()
