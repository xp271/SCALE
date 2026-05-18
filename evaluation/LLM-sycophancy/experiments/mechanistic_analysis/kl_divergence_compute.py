"""
纯计算：`plot_kl_divergence` 使用的每层 KL、pkl discovery、对齐与聚合（无 matplotlib）。
绑图参见 ``figure/mechanistic/render_kl_divergence``。

CLI 用法仍见 ``plot_kl_divergence.py``。
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path

from mcq_option_utils import EPS_DEFAULT as EPS
from mcq_option_utils import kl_divergence_probs as kl_divergence
from mcq_option_utils import logits_to_probs

# 可选：只画文件名含某关键词的模型；传空字符串表示不过滤
DEFAULT_MODEL_KEY = "Llama"
QUANT_METHODS = {"awq", "gptq", "hqq", "rtn"}


def compute_kl_per_layer(df_plain, df_opinion, max_rows=None):
    """
    按行对齐 plain 与 opinion，对每层计算平均 KL(opinion || plain)。
    返回 (layers, mean_kl_per_layer)。
    """
    if "layer_logits" not in df_plain.columns or "layer_logits" not in df_opinion.columns:
        return [], []
    n = min(len(df_plain), len(df_opinion))
    if max_rows is not None:
        n = min(n, max_rows)
    if n == 0:
        return [], []

    layer_kls = {}  # layer_idx -> list of KL values
    for idx in range(n):
        row_p = df_plain.iloc[idx]
        row_o = df_opinion.iloc[idx]
        ll_p = row_p.get("layer_logits")
        ll_o = row_o.get("layer_logits")
        if pd.isna(ll_p) or pd.isna(ll_o) or not isinstance(ll_p, dict) or not isinstance(ll_o, dict):
            continue
        for key in ll_p:
            if key not in ll_o:
                continue
            try:
                layer_idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            probs_p = logits_to_probs(ll_p[key], eps=EPS)
            probs_o = logits_to_probs(ll_o[key], eps=EPS)
            if probs_p is None or probs_o is None:
                continue
            if set(probs_p.keys()) != set(probs_o.keys()):
                continue
            kl = kl_divergence(probs_o, probs_p, eps=EPS)
            if layer_idx not in layer_kls:
                layer_kls[layer_idx] = []
            layer_kls[layer_idx].append(kl)
    if not layer_kls:
        return [], []
    layers = sorted(layer_kls.keys())
    mean_kl = [np.nanmean(layer_kls[l]) for l in layers]
    return layers, mean_kl


def compute_kl_to_full_precision_per_layer(df_quant, df_fp, max_rows=None):
    """
    按行对齐量化模型与全精度模型，对每层计算平均 KL(quantized || full_precision)。
    返回 (layers, mean_kl_per_layer)。
    """
    if "layer_logits" not in df_quant.columns or "layer_logits" not in df_fp.columns:
        return [], []
    n = min(len(df_quant), len(df_fp))
    if max_rows is not None:
        n = min(n, max_rows)
    if n == 0:
        return [], []

    layer_kls = {}  # layer_idx -> list of KL values
    for idx in range(n):
        row_q = df_quant.iloc[idx]
        row_fp = df_fp.iloc[idx]
        ll_q = row_q.get("layer_logits")
        ll_fp = row_fp.get("layer_logits")
        if pd.isna(ll_q) or pd.isna(ll_fp) or not isinstance(ll_q, dict) or not isinstance(ll_fp, dict):
            continue
        for key in ll_q:
            if key not in ll_fp:
                continue
            try:
                layer_idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            probs_q = logits_to_probs(ll_q[key], eps=EPS)
            probs_fp = logits_to_probs(ll_fp[key], eps=EPS)
            if probs_q is None or probs_fp is None:
                continue
            if set(probs_q.keys()) != set(probs_fp.keys()):
                continue
            kl = kl_divergence(probs_q, probs_fp, eps=EPS)
            if layer_idx not in layer_kls:
                layer_kls[layer_idx] = []
            layer_kls[layer_idx].append(kl)
    if not layer_kls:
        return [], []
    layers = sorted(layer_kls.keys())
    mean_kl = [np.nanmean(layer_kls[l]) for l in layers]
    return layers, mean_kl


def pkl_to_model_label(pkl_path):
    """从 pkl 文件名推出显示名，如 Llama-3.2-1B_plain_all_xxx.pkl -> Llama 3.2 1B"""
    name = Path(pkl_path).stem
    # 先去掉中间的 _logit_ / _cot_ 等推理模式标记
    for mid in ["_logit_", "_cot_"]:
        if mid in name:
            name = name.split(mid)[0]
            break
    # 再去掉末尾 _all_YYYYMMDD_HHMMSS 或 _last_...
    for suffix in ["_all_", "_last_", "_odd_", "_even_"]:
        if suffix in name:
            name = name.split(suffix)[0]
    parts = name.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        name = "_".join(parts[:-2])
    return name.replace("_", " ").replace("-", " ")


def _stem_to_model_key(p):
    """从路径 stem 提取模型键（用于对齐 plain/opinion）。"""
    stem = Path(p).stem
    stem_lower = stem.lower()
    for sep in ["_plain_", "_plain-", "_opinion_only_", "_opinion_only-", "_opinion_"]:
        if sep in stem_lower:
            idx = stem_lower.index(sep)
            stem = stem[:idx]
            break
    parts = stem.split("_")
    while len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    return "_".join(parts) if parts else stem


def find_plain_opinion_pairs(plain_paths, opinion_paths, model_key_filter=None):
    """
    按“模型名”对齐 plain 与 opinion 的 pkl。
    返回 [(plain_path, opinion_path, model_label), ...]。
    model_key_filter: 若为非空字符串，只保留文件名中含该关键词的模型；None 或空串表示不过滤。
    """
    def is_model_match(p, key):
        return key and key.lower() in Path(p).name.lower()

    plain_by_key = {}
    for p in plain_paths:
        P = Path(p)
        if not P.exists() or not P.name.lower().endswith(".pkl"):
            continue
        k = _stem_to_model_key(p)
        plain_by_key[k] = p
    opinion_by_key = {}
    for p in opinion_paths:
        P = Path(p)
        if not P.exists() or not P.name.lower().endswith(".pkl"):
            continue
        k = _stem_to_model_key(p)
        opinion_by_key[k] = p

    pairs = []
    for k in plain_by_key:
        if k not in opinion_by_key:
            continue
        if model_key_filter and not is_model_match(plain_by_key[k], model_key_filter):
            continue
        label = pkl_to_model_label(plain_by_key[k])
        pairs.append((plain_by_key[k], opinion_by_key[k], label))

    if not pairs:
        print(f"[debug] plain 文件数: {len(plain_by_key)}, 键: {list(plain_by_key.keys())}")
        print(f"[debug] opinion 文件数: {len(opinion_by_key)}, 键: {list(opinion_by_key.keys())}")
        if plain_by_key and opinion_by_key and model_key_filter:
            common = set(plain_by_key) & set(opinion_by_key)
            print(f"[debug] 共同键（未做模型过滤）: {common}")

    return pairs


def discover_quantized_fp_opinion_items(
    opinion_dir,
    dataset,
    model_id,
    method=None,
    bit=None,
    data_seed=None,
    data_seeds=None,
):
    """
    从 output_inference/{dataset}/opinion_only 自动发现
    {model_id}_{method}_{bit}_logit_all_{seed}.pkl 与 full precision 参考。
    传 method 时按 bit 分组；传 bit 时按 method 分组。
    返回 (quant_items, fp_by_seed)，quant_items 中每个元素为 (first_path, label, triples)。
    """
    opinion_dir = Path(opinion_dir).resolve()
    if not opinion_dir.exists():
        print(f"opinion_only 目录不存在: {opinion_dir}")
        return [], {}

    seed_filter = None
    if data_seeds is not None:
        seed_filter = set(data_seeds)
    elif data_seed is not None:
        seed_filter = {data_seed}

    method = method.strip().lower() if method else None
    bit = bit.strip().lower() if bit else None
    if method and bit:
        quant_pattern = re.compile(
            rf"^{re.escape(model_id)}_{re.escape(method)}_{re.escape(bit)}_logit_all_([0-9]+)$"
        )
        grouping_mode = "single"
    elif method:
        quant_pattern = re.compile(
            rf"^{re.escape(model_id)}_{re.escape(method)}_(w[0-9]+)_logit_all_([0-9]+)$"
        )
        grouping_mode = "by_bit"
    elif bit:
        quant_pattern = re.compile(rf"^{re.escape(model_id)}_(.+)_{re.escape(bit)}_logit_all_([0-9]+)$")
        grouping_mode = "by_method"
    else:
        print("自动发现量化 KL 时必须提供 --method 或 --bit。")
        return [], {}
    fp_pattern = re.compile(rf"^{re.escape(model_id)}_full_precision_logit_all_([0-9]+)$")

    fp_by_seed = {}
    quant_by_label = {}
    for p in opinion_dir.rglob("*.pkl"):
        stem = p.stem
        fp_match = fp_pattern.match(stem)
        if fp_match:
            seed = int(fp_match.group(1))
            if seed_filter is None or seed in seed_filter:
                fp_by_seed[seed] = p
            continue

        quant_match = quant_pattern.match(stem)
        if not quant_match:
            continue
        if grouping_mode == "single":
            seed = int(quant_match.group(1))
            label = f"{method.upper()} {bit}"
            sort_key = label
        elif grouping_mode == "by_bit":
            matched_bit = quant_match.group(1)
            seed = int(quant_match.group(2))
            label = matched_bit
            sort_key = matched_bit
        else:
            matched_method = quant_match.group(1)
            seed = int(quant_match.group(2))
            label = matched_method.upper() if matched_method.lower() in QUANT_METHODS else matched_method
            sort_key = matched_method.lower()
        if seed_filter is not None and seed not in seed_filter:
            continue
        quant_by_label.setdefault(sort_key, []).append((str(p), label, seed))

    quant_items = []
    for key in sorted(quant_by_label.keys()):
        triples = sorted(quant_by_label[key], key=lambda item: item[2])
        quant_items.append((triples[0][0], triples[0][1], triples))

    if not quant_items:
        print(f"未找到量化 opinion_only pkl: dataset={dataset}, model_id={model_id}, method={method}, bit={bit}")
    if not fp_by_seed:
        print(f"未找到 full precision opinion_only pkl: dataset={dataset}, model_id={model_id}")
    return quant_items, fp_by_seed


def collect_kl_divergence_curves(args: argparse.Namespace) -> tuple[list, str] | None:
    """Build ``(layers, mean_kl, label)`` series and resolved ``out_plot`` path; returns ``None`` on failure."""

    model_filter = args.model_key.strip() if args.model_key else None
    data_seed = args.data_seed
    data_seeds = args.data_seeds
    if data_seeds is not None and data_seed is not None:
        data_seed = None

    auto_discover_quant_fp = bool(args.dataset and args.model_id and (args.method or args.bit))
    kl_to_full_precision = bool(args.kl_to_full_precision or auto_discover_quant_fp)
    if kl_to_full_precision:
        if auto_discover_quant_fp:
            if args.method and args.bit:
                print("自动发现模式下请只提供 --method 或 --bit 之一：--method 画不同 bit，--bit 画不同 method。")
                return None
            opinion_dir = Path(args.output_inference_root).resolve() / args.dataset / "opinion_only"
            quant_items, fp_by_seed = discover_quantized_fp_opinion_items(
                opinion_dir,
                args.dataset,
                args.model_id,
                method=args.method,
                bit=args.bit,
                data_seed=data_seed,
                data_seeds=data_seeds,
            )

            def full_precision_opinion_for_seed(seed):
                if seed in fp_by_seed:
                    return fp_by_seed[seed]
                return None

            print(f"目录: opinion={opinion_dir} ({len(list(opinion_dir.rglob('*.pkl'))) if opinion_dir.exists() else 0} 个 pkl)")
        else:
            if not args.full_precision_opinion:
                print("启用 --kl_to_full_precision 时必须提供 --full_precision_opinion，或改用 --dataset/--model_id/--method 自动发现。")
                return None

            fp_opinion_base = Path(args.full_precision_opinion).resolve()

            def full_precision_opinion_for_seed(seed):
                if seed is None:
                    return fp_opinion_base
                suffix = f"_{seed}"
                if fp_opinion_base.stem.endswith(suffix):
                    return fp_opinion_base
                stem_parts = fp_opinion_base.stem.rsplit("_", 1)
                if len(stem_parts) == 2 and stem_parts[1].isdigit():
                    return fp_opinion_base.with_name(f"{stem_parts[0]}{suffix}{fp_opinion_base.suffix}")
                return fp_opinion_base

            def is_quant_opinion_candidate(p):
                P = Path(p)
                if not P.exists() or not P.name.lower().endswith(".pkl"):
                    return False
                if "full_precision" in P.stem.lower():
                    return False
                if P.resolve() == fp_opinion_base:
                    return False
                if model_filter and model_filter.lower() not in P.name.lower():
                    return False
                return True

            if args.opinion:
                opinion_paths = [p.strip() for p in args.opinion if p.strip()]
            elif args.opinion_dir:
                opinion_dir = Path(args.opinion_dir).resolve()
                opinion_paths = [str(p) for p in opinion_dir.rglob("*.pkl")]
                print(f"目录: opinion={opinion_dir} ({len(opinion_paths)} 个 pkl)")
            else:
                print("启用 --kl_to_full_precision 时请提供 --opinion 或 --opinion_dir，或改用 --dataset/--model_id/--method 自动发现。")
                return None

            if data_seeds is not None:
                quant_by_key = {}
                for seed in data_seeds:
                    suffix = f"_{seed}"
                    for opinion_p in opinion_paths:
                        if not Path(opinion_p).stem.endswith(suffix):
                            continue
                        if not is_quant_opinion_candidate(opinion_p):
                            continue
                        k = _stem_to_model_key(opinion_p)
                        quant_by_key.setdefault(k, []).append((opinion_p, pkl_to_model_label(opinion_p), seed))
                quant_items = []
                for k, triples in quant_by_key.items():
                    if triples:
                        quant_items.append((triples[0][0], triples[0][1], triples))
            else:
                if data_seed is not None:
                    suffix = f"_{data_seed}"
                    opinion_paths = [p for p in opinion_paths if Path(p).stem.endswith(suffix)]
                quant_items = [(p, pkl_to_model_label(p)) for p in opinion_paths if is_quant_opinion_candidate(p)]
    else:
        explicit_files = args.plain is not None and args.opinion is not None
        if explicit_files:
            plain_paths = [p.strip() for p in args.plain if p.strip()]
            opinion_paths = [p.strip() for p in args.opinion if p.strip()]
            n = min(len(plain_paths), len(opinion_paths))
            if n == 0:
                pairs = []
            else:
                pairs = [(plain_paths[i], opinion_paths[i], pkl_to_model_label(plain_paths[i])) for i in range(n)]
                if model_filter:
                    pairs = [(a, b, lbl) for a, b, lbl in pairs if model_filter.lower() in Path(a).name.lower()]
        elif args.plain_dir and args.opinion_dir:
            plain_dir = Path(args.plain_dir).resolve()
            opinion_dir = Path(args.opinion_dir).resolve()
            plain_paths = [str(p) for p in plain_dir.rglob("*.pkl")]
            opinion_paths = [str(p) for p in opinion_dir.rglob("*.pkl")]
            if data_seeds is not None:
                # 多 seed：按 seed 过滤后找 pairs，再按 model 分组，每模型对多 seed 的 KL 取平均
                seed_suffixes = [f"_{s}" for s in data_seeds]
                pairs_by_key = {}  # model_key -> [(plain_p, opinion_p), ...] 每个 seed 一对
                for suffix in seed_suffixes:
                    p_sub = [p for p in plain_paths if Path(p).stem.endswith(suffix)]
                    o_sub = [p for p in opinion_paths if Path(p).stem.endswith(suffix)]
                    sub_pairs = find_plain_opinion_pairs(p_sub, o_sub, model_key_filter=model_filter)
                    for plain_p, opinion_p, label in sub_pairs:
                        k = _stem_to_model_key(plain_p)
                        if k not in pairs_by_key:
                            pairs_by_key[k] = []
                        pairs_by_key[k].append((plain_p, opinion_p, label))
                pairs = []
                for k, triples in pairs_by_key.items():
                    if triples:
                        pairs.append((triples[0][0], triples[0][1], triples[0][2], triples))
            elif data_seed is not None:
                suffix = f"_{data_seed}"
                plain_paths = [p for p in plain_paths if Path(p).stem.endswith(suffix)]
                opinion_paths = [p for p in opinion_paths if Path(p).stem.endswith(suffix)]
                pairs = find_plain_opinion_pairs(plain_paths, opinion_paths, model_key_filter=model_filter)
                pairs = [(a, b, lbl) for a, b, lbl in pairs]
            else:
                pairs = find_plain_opinion_pairs(plain_paths, opinion_paths, model_key_filter=model_filter)
                pairs = [(a, b, lbl) for a, b, lbl in pairs]
            print(f"目录: plain={plain_dir} ({len(plain_paths)} 个 pkl), opinion={opinion_dir} ({len(opinion_paths)} 个 pkl)")
        else:
            print("请提供 --plain 与 --opinion，或 --plain_dir 与 --opinion_dir。")
            return None

    if kl_to_full_precision:
        if not quant_items:
            print("未找到可用的 quantized opinion_only pkl。请检查路径与文件名（及 --model_key）。")
            return None
    else:
        if not pairs:
            print("未找到可匹配的 (plain, opinion) 对。请检查路径与文件名（及 --model_key）。")
            return None

    start_layer = args.start_layer
    end_layer = args.end_layer

    def filter_layer_range(layers, values):
        """只保留 start_layer <= layer <= end_layer 的层。"""
        if not layers:
            return [], []
        filtered_layers, filtered_values = [], []
        for l, v in zip(layers, values):
            if l < start_layer:
                continue
            if end_layer is not None and l > end_layer:
                continue
            filtered_layers.append(l)
            filtered_values.append(v)
        return filtered_layers, filtered_values

    all_curves = []
    if kl_to_full_precision:
        for item in quant_items:
            if len(item) == 3:
                _, label, triples = item
                kl_per_seed = []
                layers_ref = None
                for opinion_p, _, seed in triples:
                    fp_opinion = full_precision_opinion_for_seed(seed)
                    if fp_opinion is None or not fp_opinion.exists():
                        print(f"跳过 {label} seed={seed}: full precision opinion pkl 不存在: {fp_opinion}")
                        continue
                    df_q = pd.read_pickle(opinion_p)
                    df_fp = pd.read_pickle(fp_opinion)
                    layers, mean_kl = compute_kl_to_full_precision_per_layer(df_q, df_fp, max_rows=args.max_rows)
                    if not layers:
                        continue
                    layers, mean_kl = filter_layer_range(layers, mean_kl)
                    if not layers:
                        continue
                    layers_ref = layers
                    kl_per_seed.append(mean_kl)
                if not kl_per_seed or layers_ref is None:
                    print(f"跳过 {label}: 无有效 layer_logits。")
                    continue
                avg_kl = np.nanmean(kl_per_seed, axis=0).tolist()
                all_curves.append((layers_ref, avg_kl, label))
            else:
                opinion_p, label = item
                fp_opinion = full_precision_opinion_for_seed(data_seed)
                if fp_opinion is None or not fp_opinion.exists():
                    print(f"跳过 {label}: full precision opinion pkl 不存在: {fp_opinion}")
                    continue
                df_q = pd.read_pickle(opinion_p)
                df_fp = pd.read_pickle(fp_opinion)
                layers, mean_kl = compute_kl_to_full_precision_per_layer(df_q, df_fp, max_rows=args.max_rows)
                if not layers:
                    print(f"跳过 {label}: 无有效 layer_logits。")
                    continue
                layers, mean_kl = filter_layer_range(layers, mean_kl)
                if not layers:
                    print(f"跳过 {label}: 在 start_layer={start_layer}, end_layer={end_layer} 范围内无层。")
                    continue
                all_curves.append((layers, mean_kl, label))
    else:
        if args.full_precision_plain and args.full_precision_opinion:
            fp_plain = Path(args.full_precision_plain).resolve()
            fp_opinion = Path(args.full_precision_opinion).resolve()
            if fp_plain.exists() and fp_opinion.exists():
                df_fp_p = pd.read_pickle(fp_plain)
                df_fp_o = pd.read_pickle(fp_opinion)
                layers_fp, mean_kl_fp = compute_kl_per_layer(df_fp_p, df_fp_o, max_rows=args.max_rows)
                if layers_fp:
                    layers_fp, mean_kl_fp = filter_layer_range(layers_fp, mean_kl_fp)
                    if layers_fp:
                        all_curves.append((layers_fp, mean_kl_fp, "Full precision"))
            else:
                print(f"警告: 全精度 pkl 不存在，跳过。plain={fp_plain}, opinion={fp_opinion}")

        for item in pairs:
            if data_seeds is not None and len(item) == 4:
                plain_p, opinion_p, label, triples = item
                kl_per_seed = []
                layers_ref = None
                for pp, op, _ in triples:
                    df_p = pd.read_pickle(pp)
                    df_o = pd.read_pickle(op)
                    layers, mean_kl = compute_kl_per_layer(df_p, df_o, max_rows=args.max_rows)
                    if not layers:
                        continue
                    layers, mean_kl = filter_layer_range(layers, mean_kl)
                    if not layers:
                        continue
                    layers_ref = layers
                    kl_per_seed.append(mean_kl)
                if not kl_per_seed or layers_ref is None:
                    print(f"跳过 {label}: 无有效 layer_logits。")
                    continue
                avg_kl = np.nanmean(kl_per_seed, axis=0).tolist()
                all_curves.append((layers_ref, avg_kl, label))
            else:
                plain_p, opinion_p, label = item[0], item[1], item[2]
                df_p = pd.read_pickle(plain_p)
                df_o = pd.read_pickle(opinion_p)
                layers, mean_kl = compute_kl_per_layer(df_p, df_o, max_rows=args.max_rows)
                if not layers:
                    print(f"跳过 {label}: 无有效 layer_logits。")
                    continue
                layers, mean_kl = filter_layer_range(layers, mean_kl)
                if not layers:
                    print(f"跳过 {label}: 在 start_layer={start_layer}, end_layer={end_layer} 范围内无层。")
                    continue
                all_curves.append((layers, mean_kl, label))
    if not all_curves:
        return None

    out_plot = args.out_plot
    if auto_discover_quant_fp and out_plot == "kl_divergence.png":
        if args.method:
            name = f"kl_quant_vs_fp_{args.dataset}_{args.model_id}_{args.method}.png"
        else:
            name = f"kl_quant_vs_fp_{args.dataset}_{args.model_id}_by_method_{args.bit}.png"
        out_plot = str(Path("figure") / name)
    if data_seed is not None and data_seeds is None:
        p = Path(out_plot)
        if not p.stem.endswith(f"_{data_seed}"):
            out_plot = str(p.parent / f"{p.stem}_{data_seed}{p.suffix}")

    return (all_curves, out_plot)
