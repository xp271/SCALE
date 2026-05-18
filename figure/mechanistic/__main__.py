"""List mechanistic plot CLIs (run from repo root with project root on PYTHONPATH)."""
from __future__ import annotations


def _usage() -> None:
    root = (
        "  python compute_decision_score.py ...   # DS / CSV (from mechanistic_analysis or import mcq_option_utils)\n"
        "  python -m figure.mechanistic.kl_plot ...\n"
        "  python -m figure.mechanistic.cli_ds_across_methods ...\n"
        "  python -m figure.mechanistic.cli_kl_across_methods ...\n"
        "  python -m figure.mechanistic.cli_ds_fp_quant ...\n"
        "  python -m figure.mechanistic.cli_ds_authority ...\n"
        "\nOr use figure.mechanistic.runners.run_decision_score / run_kl_divergence.\n"
    )
    print("Mechanistic plots / CLIs:")
    print(f"Under evaluation/LLM-sycophancy/experiments/mechanistic_analysis/: compute_* and *_compute (no thin wrappers).")
    print(root)


if __name__ == "__main__":
    _usage()
