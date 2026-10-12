"""PCA of mean-pooled prompt hidden states across user framings (compute-only). Plotting: ``authority_pca_render``.

Reads the ``layer_mean_pool_hs`` column written by ``run_syco_logit_cot.py --mean_pool_layers ...`` for the
Opinion-only prompt and the three Opinion-with-Authority levels of one model variant, fits a single
2-component PCA on all of them, and returns the projected points per framing (paper Figure 5a/b).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRAMINGS = ("opinion_only", "beginner", "intermediate", "advanced")
MEAN_POOL_COL = "layer_mean_pool_hs"


def framing_logit_pkl(root: Path, dataset: str, framing: str, model_output_name: str, data_seed: int) -> Path:
    fname = f"{model_output_name}_logit_all_{data_seed}.pkl"
    if framing == "opinion_only":
        return root / dataset / "opinion_only" / fname
    return root / dataset / "prefix_and_opinion" / "academic" / "original" / framing / fname


def _layer_key(available: set[str], layer: str) -> str:
    if layer == "last":
        return max(available, key=lambda k: int(k.split("_")[1]))
    key = f"layer_{int(layer)}"
    if key not in available:
        raise KeyError(f"{key} not saved; available: {sorted(available, key=lambda k: int(k.split('_')[1]))}")
    return key


def load_mean_pooled(pkl_path: Path, layer: str) -> tuple[np.ndarray, str]:
    """Return (n_questions, hidden_dim) mean-pooled hidden states at ``layer`` and the resolved layer key."""
    df = pd.read_pickle(pkl_path)
    if MEAN_POOL_COL not in df.columns:
        raise ValueError(
            f"{pkl_path} has no {MEAN_POOL_COL} column. Re-run run_syco_logit_cot.py with "
            "--mean_pool_layers last (delete the old pkl first; the pipeline skips existing outputs)."
        )
    cells = df[MEAN_POOL_COL].tolist()
    first = next((c for c in cells if isinstance(c, dict) and c), None)
    if first is None:
        raise ValueError(f"{pkl_path}: {MEAN_POOL_COL} is empty")
    key = _layer_key(set(first), layer)
    dim = np.asarray(first[key]).reshape(-1).shape[0]
    out = np.full((len(cells), dim), np.nan, dtype=np.float32)
    for i, c in enumerate(cells):
        if isinstance(c, dict) and key in c:
            out[i] = np.asarray(c[key], dtype=np.float32).reshape(-1)
    return out, key


def collect_authority_pca(args: argparse.Namespace):
    """Returns ``(points, explained_variance_ratio, layer_key, out_plot)``; ``points``: framing -> (n, 2)."""
    from sklearn.decomposition import PCA

    root = Path(args.output_inference_root).resolve()
    seeds = list(args.data_seeds) if args.data_seeds else [args.data_seed]
    feats: dict[str, list[np.ndarray]] = {f: [] for f in FRAMINGS}
    layer_key = None
    for seed in seeds:
        per_framing = {}
        for framing in FRAMINGS:
            p = framing_logit_pkl(root, args.dataset, framing, args.model_output_name, seed)
            if not p.exists():
                raise FileNotFoundError(f"missing {framing} logit pkl for seed={seed}: {p}")
            per_framing[framing], key = load_mean_pooled(p, args.layer)
            if layer_key is not None and key != layer_key:
                raise ValueError(f"layer mismatch: {key} vs {layer_key} ({p})")
            layer_key = key
        n = min(len(v) for v in per_framing.values())
        idx = np.arange(n)
        if args.max_rows and n > args.max_rows:
            # same questions for every framing
            idx = np.sort(np.random.default_rng(0).choice(n, args.max_rows, replace=False))
        for framing, x in per_framing.items():
            feats[framing].append(x[idx])

    stacked = {f: np.concatenate(v, axis=0) for f, v in feats.items()}
    stacked = {f: x[~np.isnan(x).any(axis=1)] for f, x in stacked.items()}
    all_x = np.concatenate([stacked[f] for f in FRAMINGS], axis=0)
    pca = PCA(n_components=2, random_state=0).fit(all_x)
    points = {f: pca.transform(stacked[f]) for f in FRAMINGS}
    print(
        f"PCA on {layer_key} ({all_x.shape[0]} points, dim {all_x.shape[1]}), "
        f"explained variance ratio = {pca.explained_variance_ratio_.round(4).tolist()}"
    )
    suffix = f"_{seeds[0]}" if len(seeds) == 1 else ""
    out_plot = args.out_plot.strip() or f"figure/pca_{args.dataset}_{args.model_output_name}{suffix}.pdf"
    return points, pca.explained_variance_ratio_, layer_key, out_plot


def build_authority_pca_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA of mean-pooled prompt hidden states: Opinion-only vs. three authority levels")
    parser.add_argument("--dataset", type=str, default="commonsenseqa")
    parser.add_argument("--output_inference_root", type=str, default="output_inference")
    parser.add_argument("--model_output_name", type=str, required=True, help="e.g. llama_3.1_8b_instruct_awq_w4 / ..._full_precision")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=None, help="Points of all listed seeds are pooled")
    parser.add_argument("--data_seed", type=int, default=42, help="Single seed (used when --data_seeds is not set)")
    parser.add_argument("--layer", type=str, default="last", help="'last' or a 0-based layer index saved in layer_mean_pool_hs")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional: random subset of questions per seed")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--out_plot", type=str, default="")
    return parser
