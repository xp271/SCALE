"""KL across methods/bit manuscript plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

QUANT_METHODS = {"awq", "gptq", "hqq", "rtn"}
FIG_SIZE = (8, 4.5)
FIG_DPI = 200
_FONT_SCALE = 1.3
AXIS_LABEL_FONTSIZE = 16 * _FONT_SCALE
AXIS_TICK_FONTSIZE = 14 * _FONT_SCALE
LEGEND_FONTSIZE = 14 * _FONT_SCALE


def _format_legend_label(raw_label: str) -> str:
    if raw_label == "full_precision":
        return "FP"
    if raw_label.startswith("w") and len(raw_label) >= 2 and raw_label[1:].isdigit():
        return f"{raw_label[1:]}-bit"
    if raw_label.lower() in QUANT_METHODS:
        return raw_label.upper()
    return str(raw_label)


def save_kl_across_methods_figure(curves: list[tuple[list, list, str, str]], *, out_plot: str) -> None:
    plt.figure(figsize=FIG_SIZE)
    x_min, x_max = None, None
    y_max = 0.0
    for layers, mean_kl, label, color in curves:
        legend_label = _format_legend_label(label)
        plt.plot(layers, mean_kl, marker="o", linestyle="-", linewidth=2, markersize=4, color=color, label=legend_label)
        lx_min, lx_max = int(np.min(layers)), int(np.max(layers))
        x_min = lx_min if x_min is None else min(x_min, lx_min)
        x_max = lx_max if x_max is None else max(x_max, lx_max)
        y_max = max(y_max, float(np.max(mean_kl)))

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    plt.ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
    plt.xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("KL Divergence", fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=FIG_DPI)
    plt.close()
    print(f"已保存: {out_plot}")
