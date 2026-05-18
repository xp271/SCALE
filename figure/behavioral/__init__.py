"""行为图：`Plain vs Opinion`（fig1）、学术权威性三档（fig2）."""

from figure.behavioral.figures import build_behavioral_arg_parser, main_cli, plot_behavioral_figures
from figure.behavioral.runners import run_fig1, run_fig2_authority

__all__ = [
    "build_behavioral_arg_parser",
    "main_cli",
    "plot_behavioral_figures",
    "run_fig1",
    "run_fig2_authority",
]
