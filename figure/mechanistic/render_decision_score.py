"""Matplotlib rendering for Decision Score layer curves (compute lives in mechanistic_analysis)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

FIG_SIZE = (8, 4.5)
FIG_DPI = 200
YLIM_DS = (0.0, 1.05)


def save_decision_score_figure(
    *,
    out_plot: str,
    title: str,
    series: list[dict[str, Any]],
    xlim: tuple[float, float] | None = None,
    dpi: float = FIG_DPI,
    figsize: tuple[float, float] = FIG_SIZE,
    ylim_ds: tuple[float, float] = YLIM_DS,
) -> None:
    """Each entry in ``series`` may include layers, values, label, color, marker, linestyle, linewidth, markersize."""
    plt.figure(figsize=figsize)
    for s in series:
        plt.plot(
            s["layers"],
            s["values"],
            color=s["color"],
            marker=s.get("marker", "o"),
            linestyle=s.get("linestyle", "-"),
            linewidth=float(s.get("linewidth", 2)),
            markersize=float(s.get("markersize", 4)),
            label=s["label"],
        )
    plt.xlabel("Layer")
    plt.ylabel("Decision Score")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.ylim(*ylim_ds)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=dpi)
    plt.close()
    print(f"Saved: {out_plot}")
