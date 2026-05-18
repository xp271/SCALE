"""手稿：KL(Opinion||Plain) across methods/bit。``python -m figure.mechanistic.cli_kl_across_methods``"""
from __future__ import annotations

import sys

from figure.mechanistic.extras.kl_across_methods_render import save_kl_across_methods_figure
from figure.mechanistic.mech_path import prepend_mech_syspath


def main() -> None:
    prepend_mech_syspath()
    from kl_divergence_across_methods_compute import build_kl_across_methods_arg_parser, collect_kl_divergence_across_methods_curves

    args = build_kl_across_methods_arg_parser().parse_args()
    packed = collect_kl_divergence_across_methods_curves(args)
    if packed is None:
        sys.exit(1)
    curves, out_plot = packed
    save_kl_across_methods_figure(curves, out_plot=out_plot)


if __name__ == "__main__":
    main()
