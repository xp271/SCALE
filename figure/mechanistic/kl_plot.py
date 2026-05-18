"""Main KL CLI (plain vs opinion / dir mode / quant vs FP). Run: ``python -m figure.mechanistic.kl_plot``.""
from __future__ import annotations

import argparse

from figure.mechanistic.mech_path import prepend_mech_syspath
from figure.mechanistic.render_kl_divergence import save_kl_divergence_figure


def build_plot_kl_arg_parser() -> argparse.ArgumentParser:
    prepend_mech_syspath()
    from kl_divergence_compute import DEFAULT_MODEL_KEY

    parser = argparse.ArgumentParser(description="Plot layer-wise KL divergence (Plain vs Opinion)")
    parser.add_argument("--plain", type=str, nargs="+", default=None, help="List of plain .pkl files")
    parser.add_argument("--opinion", type=str, nargs="+", default=None, help="List of opinion_only .pkl files")
    parser.add_argument("--plain_dir", type=str, default=None, help="Plain directory; use with --opinion_dir")
    parser.add_argument("--opinion_dir", type=str, default=None, help="opinion_only directory")
    parser.add_argument("--dataset", type=str, default=None, help="Auto-discovery: dataset name, e.g. mmlu/commonsenseqa")
    parser.add_argument("--model_id", "--model", dest="model_id", type=str, default=None, help="Auto-discovery: model id")
    parser.add_argument("--method", type=str, default=None, help="Auto-discovery: quant method, e.g. awq/gptq/hqq/rtn")
    parser.add_argument("--bit", type=str, default=None, help="Auto-discovery: quant bit width w4/w6/w8; plots different methods")
    parser.add_argument(
        "--output_inference_root",
        type=str,
        default="output_inference",
        help="Auto-discovery: output_inference root (default output_inference)",
    )
    parser.add_argument("--out_plot", type=str, default="kl_divergence.png", help="Output figure path")
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows per pkl (quick test)")
    parser.add_argument(
        "--model_key",
        type=str,
        default=DEFAULT_MODEL_KEY,
        help="Only models whose filename contains this keyword; empty string = all matches",
    )
    parser.add_argument("--data_seed", type=int, default=None, help="Data seed suffix; output figure gets _${seed} suffix")
    parser.add_argument(
        "--data_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Average per-layer KL over seeds; mutually exclusive with --data_seed",
    )
    parser.add_argument("--start_layer", type=int, default=0, help="Plot from this layer onward (0-based)")
    parser.add_argument("--end_layer", type=int, default=None, help="Plot through this layer inclusive; default last layer")
    parser.add_argument("--full_precision_plain", type=str, default=None, help="Full-precision plain .pkl; add curve if opinion matches")
    parser.add_argument("--full_precision_opinion", type=str, default=None, help="Full-precision opinion_only .pkl")
    parser.add_argument("--kl_to_full_precision", action="store_true", help="KL(quantized opinion || full-precision opinion)")
    return parser


def execute_plot_kl_divergence(args: argparse.Namespace) -> None:
    prepend_mech_syspath()
    from kl_divergence_compute import collect_kl_divergence_curves

    packed = collect_kl_divergence_curves(args)
    if packed is None:
        return
    curves, out_plot = packed
    save_kl_divergence_figure(curves, out_plot=out_plot)


def main_cli() -> None:
    parser = build_plot_kl_arg_parser()
    args = parser.parse_args()
    execute_plot_kl_divergence(args)


if __name__ == "__main__":
    main_cli()
