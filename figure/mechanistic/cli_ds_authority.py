"""Manuscript: authority three-level chosen_wrong DS. ``python -m figure.mechanistic.cli_ds_authority``"""
from __future__ import annotations

import sys

from figure.mechanistic.extras.authority_ds_render import save_authority_ds_figure
from figure.mechanistic.mech_path import prepend_mech_syspath


def main() -> None:
    prepend_mech_syspath()
    from authority_chosen_wrong_ds_compute import build_authority_arg_parser, collect_authority_ds_curves

    args = build_authority_arg_parser().parse_args()
    packed = collect_authority_ds_curves(args)
    if packed is None:
        sys.exit(1)
    curves, out_plot = packed
    save_authority_ds_figure(curves, out_plot=out_plot)


if __name__ == "__main__":
    main()
