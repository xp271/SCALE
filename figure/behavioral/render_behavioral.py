"""Matplotlib rendering for behavioral Fig1/Fig2."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from figure.behavioral.constants import (
    COLOR_ACCURACY,
    COLOR_ERROR,
    COLOR_SYCOPHANCY,
    FIG1_CORRECT_ONLY_FILENAME,
    FIG1_FILENAME,
    FIG2_FILENAME,
    LABELS_ACADEMIC_LEVEL,
    LABELS_PLAIN_OPINION,
)


def _label_segment(ax, x: float, bottom: float, height: float, pct: float, min_h: float = 0.06) -> None:
    if height >= min_h and pct > 0:
        ax.text(x, bottom + height / 2, f"{pct:.2f}%", ha="center", va="center", fontsize=9, color="black")


def _fig1_filename(
    correct_only_sr: bool,
    output_suffix: str | None,
    data_seed: int | None,
    data_seeds: list[int] | None,
) -> str:
    name1 = FIG1_CORRECT_ONLY_FILENAME if correct_only_sr else FIG1_FILENAME
    if output_suffix:
        base1, ext1 = os.path.splitext(name1)
        name1 = f"{base1}_{output_suffix}{ext1}"
    elif data_seed is not None and not data_seeds:
        base1, ext1 = os.path.splitext(name1)
        name1 = f"{base1}_{data_seed}{ext1}"
    return name1


def _fig2_filename(output_suffix: str | None, data_seed: int | None, data_seeds: list[int] | None) -> str:
    name2 = FIG2_FILENAME
    if output_suffix:
        base2, ext2 = os.path.splitext(name2)
        name2 = f"{base2}_{output_suffix}{ext2}"
    elif data_seed is not None and not data_seeds:
        base2, ext2 = os.path.splitext(name2)
        name2 = f"{base2}_{data_seed}{ext2}"
    return name2


def save_figure1(
    *,
    figure_dir: str,
    correct_only_sr: bool,
    plain: dict[str, float] | None,
    opinion: dict[str, float] | None,
    opinion_correct_only: dict[str, float] | None,
    model_label: str,
    output_suffix: str | None,
    data_seed: int | None,
    data_seeds: list[int] | None,
) -> None:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    if correct_only_sr:
        assert opinion_correct_only is not None
        x_pos = np.array([0])
        width = 0.5
        oco = opinion_correct_only
        ax1.bar(x_pos[0], oco["accuracy"], width, label="Accuracy", color=COLOR_ACCURACY)
        ax1.bar(x_pos[0], oco["sycophancy"], width, bottom=oco["accuracy"], label="Sycophancy Rate", color=COLOR_SYCOPHANCY)
        ax1.bar(
            x_pos[0],
            oco["error"],
            width,
            bottom=oco["accuracy"] + oco["sycophancy"],
            label="Error",
            color=COLOR_ERROR,
        )
        _label_segment(ax1, x_pos[0], 0, oco["accuracy"], oco["accuracy"] * 100)
        _label_segment(ax1, x_pos[0], oco["accuracy"], oco["sycophancy"], oco["sycophancy"] * 100)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(["opinion_only_correct_only"])
        ax1.set_title(f"Model Performance: Opinion-Only (Correct-Only SR) ({model_label})")
        ax1.set_xlim(-0.6, 0.6)
    else:
        assert plain is not None and opinion is not None
        x_pos = np.array([0, 1])
        width = 0.5
        ax1.bar(x_pos[0], plain["accuracy"], width, label="Accuracy", color=COLOR_ACCURACY)
        ax1.bar(x_pos[0], 0, width, bottom=plain["accuracy"], label="Sycophancy Rate", color=COLOR_SYCOPHANCY)
        ax1.bar(x_pos[0], plain["error"], width, bottom=plain["accuracy"], label="Error", color=COLOR_ERROR)
        _label_segment(ax1, x_pos[0], 0, plain["accuracy"], plain["accuracy"] * 100)
        ax1.bar(x_pos[1], opinion["accuracy"], width, color=COLOR_ACCURACY)
        ax1.bar(x_pos[1], opinion["sycophancy"], width, bottom=opinion["accuracy"], color=COLOR_SYCOPHANCY)
        ax1.bar(
            x_pos[1],
            opinion["error"],
            width,
            bottom=opinion["accuracy"] + opinion["sycophancy"],
            color=COLOR_ERROR,
        )
        _label_segment(ax1, x_pos[1], 0, opinion["accuracy"], opinion["accuracy"] * 100)
        _label_segment(ax1, x_pos[1], opinion["accuracy"], opinion["sycophancy"], opinion["sycophancy"] * 100)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(LABELS_PLAIN_OPINION)
        ax1.set_title(f"Model Performance: Plain vs. Opinion-Only Prompts ({model_label})")
        ax1.set_xlim(-0.5, 1.5)

    ax1.set_ylabel("% of Responses")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()

    os.makedirs(figure_dir, exist_ok=True)
    name1 = _fig1_filename(correct_only_sr, output_suffix, data_seed, data_seeds)
    path1 = os.path.join(figure_dir, name1)
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"Saved: {path1}")


def save_figure2(
    *,
    figure_dir: str,
    fig2_metrics: list[dict[str, float]],
    model_label: str,
    output_suffix: str | None,
    data_seed: int | None,
    data_seeds: list[int] | None,
) -> None:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(LABELS_ACADEMIC_LEVEL))
    width = 0.5
    for i, m in enumerate(fig2_metrics):
        h_s, h_a, h_e = m["sycophancy"], m["accuracy"], m["error"]
        ax2.bar(x_pos[i], h_s, width, label="Sycophancy Rate" if i == 0 else "", color=COLOR_SYCOPHANCY)
        ax2.bar(x_pos[i], h_a, width, bottom=h_s, label="Accuracy" if i == 0 else "", color=COLOR_ACCURACY)
        ax2.bar(x_pos[i], h_e, width, bottom=h_s + h_a, label="Error" if i == 0 else "", color=COLOR_ERROR)
    for i, m in enumerate(fig2_metrics):
        _label_segment(ax2, x_pos[i], 0, m["sycophancy"], m["sycophancy"] * 100)
        _label_segment(ax2, x_pos[i], m["sycophancy"], m["accuracy"], m["accuracy"] * 100)

    ax2.set_ylabel("% of Responses")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(LABELS_ACADEMIC_LEVEL)
    ax2.set_title(f"Model Performance (First-pov) ({model_label})")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(-0.5, len(LABELS_ACADEMIC_LEVEL) - 0.5)
    fig2.tight_layout()

    os.makedirs(figure_dir, exist_ok=True)
    name2 = _fig2_filename(output_suffix, data_seed, data_seeds)
    path2 = os.path.join(figure_dir, name2)
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {path2}")
