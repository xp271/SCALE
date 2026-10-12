"""Manuscript: PCA of mean-pooled prompt hidden states across user framings. ``python -m figure.mechanistic.cli_pca_authority``"""
from __future__ import annotations

from figure.mechanistic.extras.authority_pca_render import save_authority_pca_figure
from figure.mechanistic.mech_path import prepend_mech_syspath


def main() -> None:
    prepend_mech_syspath()
    from authority_pca_compute import build_authority_pca_arg_parser, collect_authority_pca

    args = build_authority_pca_arg_parser().parse_args()
    points, _, _, out_plot = collect_authority_pca(args)
    save_authority_pca_figure(points, out_plot=out_plot, legend=not args.no_legend)


if __name__ == "__main__":
    main()
