"""Scan existing output/{dataset}/plain pkl files to enumerate plottable methods.

Used by the ``--plot_scan_existing`` mode: given a save root (or LLM-sycophancy
behavioral_analysis subdir) and a ``model_id_fs``, list every ``method_id``
whose both ``plain`` and ``opinion_only`` pkl already exist for the requested
seeds.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from utils.paths import DIR_SYCO_SCRIPT


def split_method_and_seed_from_stem(stem: str, model_id_fs: str) -> tuple[str | None, int | None]:
    """从 {model_id_fs}_{method_id}[_seed].pkl 的 stem 解析 method_id 与 seed。"""
    prefix = f"{model_id_fs}_"
    if not stem.startswith(prefix):
        return None, None
    body = stem[len(prefix) :]
    m = re.match(r"^(.*)_(\d+)$", body)
    if m:
        return m.group(1), int(m.group(2))
    return body, None


def scan_existing_methods_for_plot(
    plain_dir: Path,
    opinion_dir: Path,
    model_id_fs: str,
    seeds: list[int] | None,
) -> list[str]:
    """扫描 plain_dir 下已有 pkl，返回也在 opinion_dir 内有对应文件的 method_id 列表。"""
    if not plain_dir.exists():
        return []
    methods = set()
    for p in plain_dir.glob(f"{model_id_fs}_*.pkl"):
        method_id, seed = split_method_and_seed_from_stem(p.stem, model_id_fs)
        if not method_id:
            continue
        if seeds is not None and seed not in seeds:
            continue
        # 至少要求 opinion_only 同名文件存在，避免 fig1 直接失败
        op_path = opinion_dir / p.name
        if not op_path.exists():
            continue
        methods.add(method_id)
    return sorted(methods)


def run_plot_from_existing_pkls(
    *,
    syco_repo: Path,
    dataset: str,
    model_id_fs: str,
    seeds_for_plot: list[int],
    figure_dir: str = "figure",
    correct_only_sr: bool = True,
    behavioral_output_base: str | None = None,
) -> int:
    """只基于已有 pkl 批量绘图（不跑量化/评测）。

    ``behavioral_output_base`` 为 None 时按旧布局扫描
    ``LLM-sycophancy/experiments/behavioral_analysis/output/{dataset}/...``；
    显式给出时（新布局 save_root/behavioral）按其作为扫描根。
    """
    ba_dir = syco_repo / DIR_SYCO_SCRIPT
    if behavioral_output_base:
        base = Path(behavioral_output_base)
    else:
        base = ba_dir / "output"
    plain_dir = base / dataset / "plain"
    opinion_dir = base / dataset / "opinion_only"
    methods = scan_existing_methods_for_plot(plain_dir, opinion_dir, model_id_fs, seeds_for_plot if seeds_for_plot else None)
    if not methods:
        print(
            f"[PlotOnly] no eligible methods found in {plain_dir} for model_id_fs={model_id_fs}",
            file=sys.stderr,
        )
        return 0
    ok_cnt = 0
    for method_id in methods:
        plot_output_name = f"{model_id_fs}_{method_id}"
        out_suffix = f"{dataset}_{plot_output_name}" + ("_correct_only" if correct_only_sr else "")
        cmd = [
            sys.executable,
            "plot_figure2.py",
            "--which", "fig1",
            "--output_base", str(base) if behavioral_output_base else "output",
            "--dataset_subdir", dataset,
            "--figure_dir", figure_dir,
            "--model_type", plot_output_name,
            "--output_suffix", out_suffix,
            "--data_seeds",
            *[str(s) for s in seeds_for_plot],
        ]
        if correct_only_sr:
            cmd.extend(["--correct_only_sr", "--baseline_model_type", f"{model_id_fs}_full_precision"])
        print(f"[PlotOnly] plotting {plot_output_name} (seeds={seeds_for_plot}) ...")
        ret = subprocess.run(cmd, cwd=str(ba_dir), timeout=300)
        if ret.returncode == 0:
            ok_cnt += 1
    print(f"[PlotOnly] done: {ok_cnt}/{len(methods)} methods plotted.")
    return ok_cnt
