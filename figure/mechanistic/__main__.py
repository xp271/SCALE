"""列出机理绘图相关 CLI（均在仓库根目录、PYTHONPATH 含项目根时运行）。"""
from __future__ import annotations


def _usage() -> None:
    root = (
        "  python compute_decision_score.py ...   # DS / CSV（在 mechanistic_analysis 目录或保证可 import mcq_option_utils）\n"
        "  python -m figure.mechanistic.kl_plot ...\n"
        "  python -m figure.mechanistic.cli_ds_across_methods ...\n"
        "  python -m figure.mechanistic.cli_kl_across_methods ...\n"
        "  python -m figure.mechanistic.cli_ds_fp_quant ...\n"
        "  python -m figure.mechanistic.cli_ds_authority ...\n"
        "\n或使用 figure.mechanistic.runners.run_decision_score / run_kl_divergence。\n"
    )
    print("Mechanistic plots / CLIs:")
    print(f"evaluation/LLM-sycophancy/experiments/mechanistic_analysis/ 下有 compute_* 与各 *_compute（无薄封装脚本）。")
    print(root)


if __name__ == "__main__":
    _usage()
