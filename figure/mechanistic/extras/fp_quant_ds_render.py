"""FP vs quantized four-line Decision Score plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_SIZE = (8, 4.5)
FIG_DPI = 200
YLIM_DS = (0.0, 1.05)
_FONT_SCALE = 1.3
AXIS_LABEL_FONTSIZE = 16 * _FONT_SCALE
AXIS_TICK_FONTSIZE = 14 * _FONT_SCALE
LEGEND_FONTSIZE = 14 * _FONT_SCALE


def save_fp_quant_four_line_ds(
    series: list[tuple[np.ndarray, np.ndarray, str, str, str]],
    *,
    out_path: Path,
) -> None:
    plt.figure(figsize=FIG_SIZE)
    x_min, x_max = None, None
    for layers, ds_vals, label, color, linestyle in series:
        plt.plot(
            layers,
            ds_vals,
            marker="o",
            linestyle=linestyle,
            linewidth=2,
            markersize=4,
            color=color,
            label=label,
        )
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")
