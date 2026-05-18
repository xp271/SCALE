"""Manuscript: FP vs quant four DS curves (correct / chosen wrong). ``python -m figure.mechanistic.cli_ds_fp_quant``"""
from __future__ import annotations

import sys

from figure.mechanistic.extras.fp_quant_ds_render import save_fp_quant_four_line_ds
from figure.mechanistic.mech_path import prepend_mech_syspath


def main() -> None:
    prepend_mech_syspath()
    from decision_score_fp_quant_plain_opinion_compute import build_fp_quant_arg_parser, collect_fp_quant_ds_series

    args = build_fp_quant_arg_parser().parse_args()
    packed = collect_fp_quant_ds_series(args)
    if packed is None:
        sys.exit(1)
    series_correct, series_wrong, out_c, out_w = packed
    save_fp_quant_four_line_ds(series_correct, out_path=out_c)
    save_fp_quant_four_line_ds(series_wrong, out_path=out_w)


if __name__ == "__main__":
    main()
