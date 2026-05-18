"""Filename, colors, and output subdir semantics (aligned with run_syco layout)."""
from __future__ import annotations

FIG1_FILENAME = "fig2_plain_vs_opinion.png"
FIG2_FILENAME = "fig2_advanced_first_pov.png"
FIG1_CORRECT_ONLY_FILENAME = "fig2_opinion_correct_only.png"

COLOR_ACCURACY = "#7eb8da"
COLOR_SYCOPHANCY = "#f4a582"
COLOR_ERROR = "#e74c3c"

LABELS_PLAIN_OPINION = ["plain", "opinion_only"]
LABELS_ACADEMIC_LEVEL = ["Beginner", "Intermediate", "Advanced"]


def output_dirs_for_dataset(dataset_subdir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Subdirs per condition (relative to ``output_base``)."""
    d = (dataset_subdir or "mmlu").strip().strip("/\\") or "mmlu"
    dirs_fig1 = {
        "plain": f"{d}/plain",
        "opinion_only": f"{d}/opinion_only",
    }
    dirs_fig2 = {
        "beginner": f"{d}/prefix_and_opinion/academic/original/beginner",
        "intermediate": f"{d}/prefix_and_opinion/academic/original/intermediate",
        "advanced": f"{d}/prefix_and_opinion/academic/original/advanced",
    }
    return dirs_fig1, dirs_fig2
