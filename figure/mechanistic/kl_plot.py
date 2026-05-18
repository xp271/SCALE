"""主 KL（plain vs opinion / 目录模式 / quant vs FP）CLI。运行：``python -m figure.mechanistic.kl_plot``。"""
from __future__ import annotations

import argparse

from figure.mechanistic.mech_path import prepend_mech_syspath
from figure.mechanistic.render_kl_divergence import save_kl_divergence_figure


def build_plot_kl_arg_parser() -> argparse.ArgumentParser:
    prepend_mech_syspath()
    from kl_divergence_compute import DEFAULT_MODEL_KEY

    parser = argparse.ArgumentParser(description="画 Layer-wise KL Divergence（Plain vs Opinion）")
    parser.add_argument("--plain", type=str, nargs="+", default=None, help="plain 的 .pkl 列表")
    parser.add_argument("--opinion", type=str, nargs="+", default=None, help="opinion_only 的 .pkl 列表")
    parser.add_argument("--plain_dir", type=str, default=None, help="plain 目录，与 --opinion_dir 成对使用")
    parser.add_argument("--opinion_dir", type=str, default=None, help="opinion_only 目录")
    parser.add_argument("--dataset", type=str, default=None, help="自动发现模式：数据集名，如 mmlu/commonsenseqa")
    parser.add_argument("--model_id", "--model", dest="model_id", type=str, default=None, help="自动发现模式：模型标识")
    parser.add_argument("--method", type=str, default=None, help="自动发现模式：量化方法，如 awq/gptq/hqq/rtn")
    parser.add_argument("--bit", type=str, default=None, help="自动发现模式：量化精度，如 w4/w6/w8；传入后绘制不同 method")
    parser.add_argument(
        "--output_inference_root",
        type=str,
        default="output_inference",
        help="自动发现模式：output_inference 根目录，默认 output_inference",
    )
    parser.add_argument("--out_plot", type=str, default="kl_divergence.png", help="输出图路径")
    parser.add_argument("--max_rows", type=int, default=None, help="每个 pkl 最多用多少行（用于快速测试）")
    parser.add_argument(
        "--model_key",
        type=str,
        default=DEFAULT_MODEL_KEY,
        help="只画文件名中含该关键词的模型；传空串画所有匹配",
    )
    parser.add_argument("--data_seed", type=int, default=None, help="数据种子后缀；输出图会带 _${seed} 后缀")
    parser.add_argument(
        "--data_seeds",
        type=int,
        nargs="+",
        default=None,
        help="多个种子时对每层 KL 取平均；与 --data_seed 二选一",
    )
    parser.add_argument("--start_layer", type=int, default=0, help="只画该层及之后（0-based）")
    parser.add_argument("--end_layer", type=int, default=None, help="只画到该层（含）；不传则到最后一层")
    parser.add_argument("--full_precision_plain", type=str, default=None, help="全精度 plain .pkl；与 opinion 同上则加曲线")
    parser.add_argument("--full_precision_opinion", type=str, default=None, help="全精度 opinion_only .pkl")
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
