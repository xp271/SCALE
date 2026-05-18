"""Manuscript: chosen_wrong DS (same bit / different methods or vice versa). ``python -m figure.mechanistic.cli_ds_across_methods``"""
from __future__ import annotations

import sys

from figure.mechanistic.extras.ds_across_methods_render import save_ds_across_methods_figure
from figure.mechanistic.mech_path import prepend_mech_syspath


def main() -> None:
    prepend_mech_syspath()
    from decision_score_across_methods_compute import build_ds_across_methods_arg_parser, collect_decision_score_across_methods_curves

    args = build_ds_across_methods_arg_parser().parse_args()
    packed = collect_decision_score_across_methods_curves(args)
    if packed is None:
        sys.exit(1)
    curves, out_plot, compare_mode = packed
    save_ds_across_methods_figure(curves, out_plot=out_plot, compare_mode=compare_mode)


if __name__ == "__main__":
    main()
