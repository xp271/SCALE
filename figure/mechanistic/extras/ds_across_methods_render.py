"""Matplotlib rendering for DS across methods/bit (chosen wrong)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_QUANT_METHODS = {"awq", "gptq", "hqq", "rtn"}
FIG_SIZE = (8, 4.5)
FIG_DPI = 200
YLIM_DS = (0.0, 1.05)
_FONT_SCALE = 1.3
AXIS_LABEL_FONTSIZE = 16 * _FONT_SCALE
AXIS_TICK_FONTSIZE = 14 * _FONT_SCALE
LEGEND_FONTSIZE = 14 * _FONT_SCALE


def _format_legend_label(raw_label: str, compare_mode: str) -> str:
    if raw_label == "full_precision":
        return "FP"
    if compare_mode == "by_bit" and raw_label.startswith("w") and raw_label[1:].isdigit():
        return f"{raw_label[1:]}-bit"
    if raw_label.lower() in _QUANT_METHODS:
        return raw_label.upper()
    return raw_label


def save_ds_across_methods_figure(curves: list, *, out_plot: str, compare_mode: str) -> None:
    """``curves``: list of (layers, ds_vals, raw_label, color)."""
    plt.figure(figsize=FIG_SIZE)
    x_min, x_max = None, None
    for layers, ds_vals, label, color in curves:
        legend_label = _format_legend_label(label, compare_mode)
        plt.plot(layers, ds_vals, marker="o", linestyle="-", linewidth=2, markersize=4, color=color, label=legend_label)
        lx_min, lx_max = int(np.min(layers)), int(np.max(layers))
        x_min = lx_min if x_min is None else min(x_min, lx_min)
        x_max = lx_max if x_max is None else max(x_max, lx_max)

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    plt.ylim(*YLIM_DS)
    plt.xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Decision Score", fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=FIG_DPI)
    plt.close()
    print(f"已保存: {out_plot}")
