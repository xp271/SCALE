"""Matplotlib rendering for layer-wise KL curves (numeric collection in mechanistic_analysis)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

QUANT_METHODS = {"awq", "gptq", "hqq", "rtn"}

FIG_SIZE = (8, 4.5)
FIG_DPI = 200
_FONT_SCALE = 1.3
AXIS_LABEL_FONTSIZE = 16 * _FONT_SCALE
AXIS_TICK_FONTSIZE = 14 * _FONT_SCALE
LEGEND_FONTSIZE = 14 * _FONT_SCALE


def format_quant_method_legend_label(label):
    parts = str(label).split()
    if len(parts) == 1 and parts[0].startswith("w") and parts[0][1:].isdigit():
        return f"{parts[0][1:]}-bit"
    return " ".join(part.upper() if part.lower() in QUANT_METHODS else part for part in parts)


def save_kl_divergence_figure(
    all_curves: list[tuple[list, list, str]],
    *,
    out_plot: str,
    dpi: float = FIG_DPI,
    figsize: tuple[float, float] = FIG_SIZE,
) -> None:
    if not all_curves:
        return
    distinct_colors = ["#333333", "#0066cc", "#cc0000", "#008833", "#9944aa", "#dd8800", "#00aacc"]
    fig, ax = plt.subplots(figsize=figsize)
    for i, (layers, mean_kl, label) in enumerate(all_curves):
        color = distinct_colors[i % len(distinct_colors)]
        legend_label = format_quant_method_legend_label(label)
        ax.plot(
            layers,
            mean_kl,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=4,
            color=color,
            label=legend_label,
        )
    ax.set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("KL Divergence", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    y_max = max(max(kl) for _, kl, _ in all_curves)
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
    fig.tight_layout()
    out_parent = Path(out_plot).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_plot}")
