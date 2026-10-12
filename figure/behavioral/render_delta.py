"""Matplotlib rendering of Delta_OSR (Figure 1) and Delta_ASR (Figure 2); legends are separate files."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# label, color, marker
METHOD_STYLES = {
    "rtn": ("RTN", "#2a78d6", "o"),
    "gptq": ("GPTQ", "#eb6834", "s"),
    "awq": ("AWQ", "#1baf7a", "D"),
    "hqq": ("HQQ", "#d55181", "^"),
}
LEVEL_STYLES = {
    "beginner": ("Beginner", "#2a78d6"),
    "intermediate": ("Intermediate", "#eb6834"),
    "advanced": ("Advanced", "#1baf7a"),
}
GRID_COLOR = "#e5e5e5"
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
OSR_FIGSIZE = (2.4, 1.75)
OSR_FONTSIZE = 11
ASR_FIGSIZE = (3.2, 2.3)
ASR_FONTSIZE = 16
ASR_DEFAULT_YLIM = (-5.0, 5.0)
BAR_WIDTH = 0.25


def _method_style(method: str) -> tuple[str, str, str]:
    return METHOD_STYLES.get(method, (method.upper(), "#555555", "o"))


def _save(fig, out_plot: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_plot}")


def save_opinion_delta_figure(
    series: dict[str, list[float]],
    *,
    bits: list[int],
    out_plot: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """``series``: method -> Delta_OSR (%) per entry of ``bits``; NaN points are left out."""
    x = np.arange(len(bits))
    with plt.rc_context({**RC, "font.size": OSR_FONTSIZE}):
        fig, ax = plt.subplots(figsize=OSR_FIGSIZE)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
        for method, values in series.items():
            label, color, marker = _method_style(method)
            ax.plot(x, values, color=color, marker=marker, linewidth=2, markersize=4, label=label, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bits])
        ax.set_xlabel("Bit-width")
        ax.set_ylabel(r"$\Delta_{\mathrm{OSR}}$ (%)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        _save(fig, out_plot)


def save_authority_delta_figure(
    series: dict[str, list[float]],
    *,
    bits: list[int],
    out_plot: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """``series``: level -> Delta_ASR (%) per entry of ``bits`` (grouped bars, one group per bit-width)."""
    x = np.arange(len(bits))
    n = len(series)
    offsets = (np.arange(n) - (n - 1) / 2) * BAR_WIDTH
    values_all = [v for vals in series.values() for v in vals if np.isfinite(v)]
    if ylim is None:
        lim = max([ASR_DEFAULT_YLIM[1]] + [abs(v) * 1.1 for v in values_all])
        ylim = (-lim, lim)
    with plt.rc_context({**RC, "font.size": ASR_FONTSIZE}):
        fig, ax = plt.subplots(figsize=ASR_FIGSIZE)
        for off, (level, values) in zip(offsets, series.items()):
            label, color = LEVEL_STYLES.get(level, (level.capitalize(), "#555555"))
            ax.bar(x + off, np.nan_to_num(values, nan=0.0), width=BAR_WIDTH, color=color, label=label, zorder=3)
        ax.axhline(0, color="black", linewidth=1, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bits])
        ax.set_xlabel("Bit-width")
        ax.set_ylabel(r"$\Delta_{\mathrm{ASR}}$ (%)")
        ax.set_ylim(*ylim)
        if tuple(ylim) == ASR_DEFAULT_YLIM:
            ax.set_yticks([-5, -2.5, 0, 2.5, 5])
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        _save(fig, out_plot)


def save_legends(figure_dir: str, *, methods: list[str], with_authority: bool = True) -> None:
    """Horizontal legends shared by all panels: opinion_delta_sr_legend.pdf / authority_delta_sr_legend.pdf."""
    with plt.rc_context({**RC, "font.size": OSR_FONTSIZE}):
        handles = []
        for m in methods:
            label, color, marker = _method_style(m)
            handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=2, markersize=4, label=label))
        fig = plt.figure(figsize=(4, 0.3))
        fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
        _save(fig, os.path.join(figure_dir, "opinion_delta_sr_legend.pdf"))
        if with_authority:
            handles = [Patch(color=c, label=l) for l, c in LEVEL_STYLES.values()]
            fig = plt.figure(figsize=(4, 0.3))
            fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
            _save(fig, os.path.join(figure_dir, "authority_delta_sr_legend.pdf"))
