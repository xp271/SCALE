"""Authority three-level Decision Score curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEVEL_COLORS = {
    "beginner": "#1f77b4",
    "intermediate": "#ff7f0e",
    "advanced": "#2ca02c",
}


def _level_legend_label(level: str) -> str:
    return (level or "").strip().capitalize()


FIG_SIZE = (8, 4.5)
FIG_DPI = 200
YLIM_DS = (0.0, 1.05)
_FONT_SCALE = 1.3
AXIS_LABEL_FONTSIZE = 16 * _FONT_SCALE
AXIS_TICK_FONTSIZE = 14 * _FONT_SCALE
LEGEND_FONTSIZE = 14 * _FONT_SCALE


def save_authority_ds_figure(curves: list[tuple[str, np.ndarray, np.ndarray]], *, out_plot: str) -> None:
    plt.figure(figsize=FIG_SIZE)
    x_min, x_max = None, None
    for level, layers, ds_vals in curves:
        plt.plot(
            layers,
            ds_vals,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=4,
            color=LEVEL_COLORS[level],
            label=_level_legend_label(level),
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
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_plot}")
