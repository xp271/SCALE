"""PCA scatter of user framings (Opinion-only vs. three authority levels); no in-plot text annotations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# framing -> (legend label, color); colors as in the paper caption of Figure 5
FRAMING_STYLES = {
    "opinion_only": ("Opinion-only", "#92e0df"),
    "beginner": ("Beginner", "#e96658"),
    "intermediate": ("Intermediate", "#44b976"),
    "advanced": ("Advanced", "#3b579c"),
}
FIG_SIZE = (6, 4.8)
FIG_DPI = 200
FONT_SIZE = 12
POINT_SIZE = 10
POINT_ALPHA = 0.7
ELLIPSE_N_STD = 2.0
ELLIPSE_ALPHA = 0.2


def _cov_ellipse(xy: np.ndarray, color: str) -> Ellipse | None:
    if len(xy) < 3:
        return None
    cov = np.cov(xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * ELLIPSE_N_STD * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy.mean(axis=0), width, height, angle=angle, facecolor=color, edgecolor=color,
                   alpha=ELLIPSE_ALPHA, linewidth=1, zorder=1)


def save_authority_pca_figure(points: dict[str, np.ndarray], *, out_plot: str, legend: bool = True) -> None:
    """``points``: framing -> (n, 2) PCA coordinates, drawn in FRAMING_STYLES order."""
    with plt.rc_context({"font.size": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for z, (framing, (label, color)) in enumerate(FRAMING_STYLES.items()):
            xy = points.get(framing)
            if xy is None or len(xy) == 0:
                continue
            ax.scatter(xy[:, 0], xy[:, 1], s=POINT_SIZE, color=color, alpha=POINT_ALPHA, linewidths=0,
                       label=label, rasterized=True, zorder=2 + z)
            ell = _cov_ellipse(xy, color)
            if ell is not None:
                ax.add_patch(ell)
            cx, cy = xy.mean(axis=0)
            ax.scatter([cx], [cy], marker="*", s=160, color=color, edgecolors="black", linewidths=0.8, zorder=10)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5, zorder=0)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        if legend:
            ax.legend(loc="upper right", markerscale=2, frameon=True)
        fig.tight_layout()
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=FIG_DPI)
        plt.close(fig)
    print(f"Saved: {out_plot}")
