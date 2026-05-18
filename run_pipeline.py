#!/usr/bin/env python3
"""CLI-driven thin entry: LightCompress -> LLM-sycophancy eval -> figures.

CLI 必填：``--dataset``、``--model``、``--method``、``--bits``、``--gpu``；跑评估时还需 ``--eval``
（``--skip_eval`` 时可省略 ``--eval``）。其余默认见 [config/pipeline_config.yaml](config/pipeline_config.yaml)。
Use ``--plot_scan_existing`` to skip quant/eval and re-plot from existing pkls.

Usage:
    python run_pipeline.py \\
        --dataset mmlu \\
        --model mistral_7b_instruct_v0_3 \\
        --method Awq \\
        --bits 4 \\
        --eval behavioral,mechanistic \\
        --gpu cuda:0 \\
        [--config config/pipeline_config.yaml]

    behavioral 图默认 full SR；若需 correct-only，在 ``--eval`` 中加元 token ``correct_only``（须同时选
    ``behavioral`` 等 job token），例如 ``behavioral,correct_only``。若 yaml 里开了 correct-only，可用 ``full_sr`` 强制全量 SR（与 ``correct_only`` 互斥）。

    多 seed 评估与绘图平均：默认 ``--eval_avg_runs 3``、``--data_seed_rng 42``，由 PRNG 生成 3 个 ``data_seed``，
    数据准备 / 评估 / 制图会对这 3 套数据各跑一遍，图里对多份 pkl 求平均。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from config.parser import (
    CliArgs,
    eval_tags_for_jobs,
    expand_eval_modes,
    filter_eval_jobs_by_tags,
    narrow_config_for_cli,
    normalize_gpu,
    parse_cli,
)
from evaluation.data_prep import ensure_syco_data_for_seed, verify_academic_three_levels
from evaluation.job_builder import build_syco_eval_jobs
from evaluation.runner import eval_model_against_jobs
from figure.orchestrator import run_plot_phase
from figure.scan import run_plot_from_existing_pkls
from quantization.registry import get_method, resolve_method_quant_combo
from quantization.runner import run_lightcompress
from quantization.yml_builder import dump_run_yml
from utils.config_loader import load_config
from utils.env import setup_hf_cache_env
from utils.paths import (
    DIR_LIGHTCOMPRESS,
    DIR_LLM_SYCOPHANCY,
    DIR_SYCO_SCRIPT,
    fs_safe_label,
    resolve_path,
    resolve_path_roots,
)
from utils.seeds import generate_eval_data_seeds, normalize_data_seeds


def _resolve_repos(cfg: dict, config_dir: Path, script_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve (llmc_root, syco_repo, syco_script_dir)."""
    llmc_root = resolve_path(cfg.get("llmc_dir"), config_dir) or (script_dir / DIR_LIGHTCOMPRESS)
    syco_repo = resolve_path(cfg.get("syco_repo_dir"), config_dir) or (script_dir / DIR_LLM_SYCOPHANCY)
    return llmc_root, syco_repo, syco_repo / DIR_SYCO_SCRIPT


def _handle_plot_scan_mode(
    cli: CliArgs,
    cfg: dict,
    *,
    syco_repo: Path,
    save_root: Path,
    result_root: Path,
) -> None:
    """``--plot_scan_existing`` 短路：只基于已有 pkl 绘图。"""
    syco_cfg = cfg.get("syco", {}) or {}
    if cli.dataset:
        scan_dataset = cli.dataset
    else:
        datasets = syco_cfg.get("datasets") or {}
        if not datasets:
            print("plot-only 模式下未指定 --dataset，且 yaml syco.datasets 为空。", file=sys.stderr)
            sys.exit(2)
        scan_dataset = next(iter(datasets))
    assert cli.plot_scan_model_id is not None
    scan_model_id_fs = fs_safe_label(cli.plot_scan_model_id)
    if cli.plot_scan_seeds:
        scan_seeds = [int(x.strip()) for x in cli.plot_scan_seeds.split(",") if x.strip()]
    else:
        scan_seeds = generate_eval_data_seeds(cli.data_seed_rng, cli.eval_avg_runs)
    scan_figure_dir = cli.plot_scan_figure_dir or str(result_root / "behavioral" / scan_dataset)
    if cli.plot_scan_correct_only is None:
        scan_correct_only = bool(syco_cfg.get("eval_sr_correct_only", False))
    else:
        scan_correct_only = str(cli.plot_scan_correct_only).strip().lower() in ("1", "true", "yes", "on")
    behavioral_base = str(save_root / "behavioral") if save_root else None
    run_plot_from_existing_pkls(
        syco_repo=syco_repo,
        dataset=scan_dataset,
        model_id_fs=scan_model_id_fs,
        seeds_for_plot=scan_seeds,
        figure_dir=scan_figure_dir,
        correct_only_sr=scan_correct_only,
        behavioral_output_base=behavioral_base,
    )


def _quantize_one(
    *,
    model_id: str,
    model_id_fs: str,
    model_path_cfg: str,
    model_type: str,
    method_name: str,
    method_id: str,
    weight_bits: int,
    llmc_root: Path,
    cache_root: Path,
    calib_root: Path,
    eval_root: Path,
    calib_auto_download: bool,
    eval_auto_download: bool,
    base_seed: int,
    nproc: int,
    cuda_visible: str | None,
    run_yml_dir: Path,
) -> Path | None:
    """Quantize one (model, method, bits) tuple; return fake_quant_dir on success."""
    method = get_method(method_name)
    if method is None:
        print(f"Unknown method: {method_name}", file=sys.stderr)
        return None
    template_path = method.template_path(llmc_root)
    if not template_path.exists():
        print(f"Template yml not found: {template_path}", file=sys.stderr)
        return None
    save_path = cache_root / model_id_fs / method_id
    fake_quant_dir = save_path / "fake_quant_model"
    if fake_quant_dir.exists():
        if any(fake_quant_dir.iterdir()):
            print(f"[Skip Quant] {model_id} x {method_id}: fake_quant_model already exists at {fake_quant_dir}")
            return fake_quant_dir
        try:
            fake_quant_dir.rmdir()
        except OSError as e:
            print(f"Failed to remove empty fake_quant dir {fake_quant_dir}: {e}", file=sys.stderr)
            return None
    task_id = f"{model_id_fs}_{method_id}_{int(time.time())}"
    run_yml = method.build_run_yml(
        llmc_root,
        model_path=model_path_cfg,
        model_type=model_type,
        save_path=save_path,
        calib_root=calib_root,
        eval_root=eval_root,
        base_seed=base_seed,
        weight_bits=weight_bits,
        calib_auto_download=calib_auto_download,
        eval_auto_download=eval_auto_download,
    )
    run_yml_path = run_yml_dir / f"{task_id}.yaml"
    dump_run_yml(run_yml, run_yml_path)
    vis_note = f"CUDA_VISIBLE_DEVICES={cuda_visible}" if cuda_visible is not None else "CUDA_VISIBLE_DEVICES unset"
    print(f"[Quant] {model_id} x {method_id} (task_id={task_id}, {vis_note})")
    print(f"  -> weight_bits={weight_bits}")
    ok = run_lightcompress(llmc_root, run_yml_path, task_id, nproc, cuda_devices=cuda_visible)
    if not ok:
        print(f"LightCompress failed for {model_id} x {method_id}", file=sys.stderr)
        return None
    if not fake_quant_dir.exists():
        print(f"fake_quant_model not found at {fake_quant_dir}", file=sys.stderr)
        return None
    return fake_quant_dir


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    cli = parse_cli()
    config_path = cli.config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg_raw = load_config(config_path)
    cfg = narrow_config_for_cli(cfg_raw, cli)
    config_dir = config_path.parent

    cache_root, save_root, result_root = resolve_path_roots(cfg, config_dir)
    calib_root = resolve_path(cfg.get("calib_root"), config_dir) or config_dir
    eval_root = resolve_path(cfg.get("eval_root"), config_dir) or config_dir
    calib_auto_download = bool(cfg.get("calib_auto_download", False))
    eval_auto_download = bool(cfg.get("eval_auto_download", False))
    setup_hf_cache_env(cache_root)

    llmc_root, syco_repo, syco_script_dir = _resolve_repos(cfg, config_dir, script_dir)
    if not syco_script_dir.exists():
        print(f"LLM-sycophancy experiments not found at {syco_script_dir}. Set config 'syco_repo_dir'.", file=sys.stderr)
        sys.exit(1)

    # 1) plot-only 短路
    if cli.plot_scan_existing:
        _handle_plot_scan_mode(
            cli, cfg,
            syco_repo=syco_repo,
            save_root=save_root,
            result_root=result_root,
        )
        return

    if not llmc_root.exists():
        print(f"LightCompress not found at {llmc_root}. Set config 'llmc_dir'.", file=sys.stderr)
        sys.exit(1)

    syco_args = cfg.get("syco", {}) or {}
    syco_dataset = syco_args.get("dataset")
    data_slug = syco_args.get("data_slug") or syco_dataset
    eval_tags = expand_eval_modes(cli.eval_modes)
    eval_job_tags = eval_tags_for_jobs(eval_tags)
    eval_mechanistic = bool(syco_args.get("eval_mechanistic", False))
    eval_authority_advanced = bool(syco_args.get("eval_authority_advanced", False))
    eval_behavior_prefix = bool(syco_args.get("eval_behavior_prefix", False))
    eval_sr_correct_only = bool(syco_args.get("eval_sr_correct_only", False))

    # CLI 决定的「最大集合」jobs，再按 expanded tags 过滤
    eval_jobs: list[dict] = []
    if not cli.skip_eval:
        eval_jobs_all = build_syco_eval_jobs(
            data_slug,
            eval_mechanistic=eval_mechanistic,
            eval_authority_advanced=eval_authority_advanced,
            eval_behavior_prefix=eval_behavior_prefix,
            behavior_input_filename=syco_args.get("behavior_input_filename"),
        )
        eval_jobs = filter_eval_jobs_by_tags(eval_jobs_all, eval_job_tags)
        if not eval_jobs:
            print(
                f"--eval {cli.eval_modes} 用于 job 过滤的标签为 {sorted(eval_job_tags)} 后无可执行的 job。请检查 token 拼写。",
                file=sys.stderr,
            )
            sys.exit(2)

    nproc = cfg.get("nproc_per_node", 1)
    quant_cuda_visible = normalize_gpu(cli.gpu) if cli.gpu else None
    # CUDA_VISIBLE_DEVICES 限定时子进程内首张卡为 cuda:0
    syco_device = "cuda:0" if quant_cuda_visible is not None else syco_args.get("device", "cuda:0")
    base_seed = 42
    aggregate_csv = cfg.get("aggregate_accuracy_csv")
    plot_figures = bool(cfg.get("plot_figures", True))
    figure_output_dir = cfg.get("figure_output_dir")

    data_seeds = normalize_data_seeds(syco_args.get("data_seed", 42))
    if not cli.skip_eval:
        seeds_nonone = [s for s in data_seeds if s is not None]
        if seeds_nonone:
            print(
                f"[data_seed] --data_seed_rng={cli.data_seed_rng} --eval_avg_runs={cli.eval_avg_runs} "
                f"→ 本次评估 {len(seeds_nonone)} 个数据集种子: {seeds_nonone}"
            )

    # 2) 确保所需 seed 的 lib/*.pkl 与 Academic 三档输入齐备（仅评估阶段需要）
    if not cli.skip_eval:
        for s in data_seeds:
            if s is None:
                continue
            if not ensure_syco_data_for_seed(syco_repo, s, syco_args):
                print(f"无法为 data_seed={s} 准备数据，退出", file=sys.stderr)
                sys.exit(1)
            if eval_authority_advanced and not verify_academic_three_levels(syco_repo, data_slug, s):
                sys.exit(1)

    run_yml_dir = cache_root / "_run_configs"
    run_yml_dir.mkdir(parents=True, exist_ok=True)

    behavioral_base = str((save_root / "behavioral").resolve())
    mechanistic_base = str((save_root / "mechanistic").resolve())
    output_base_map = {"behavioral": behavioral_base, "mechanistic": mechanistic_base}

    models = cfg.get("models") or []
    methods = cfg.get("methods") or []
    if not models or not methods:
        print("窄化后的 cfg 缺少 models / methods。退出。", file=sys.stderr)
        sys.exit(2)

    accuracy_rows: list[dict] = []
    completed_combos: list[tuple[str, str]] = []

    # 3) FP 评估：先对该模型在 full_precision 下跑一遍（skip_eval 时直接跳）
    model = models[0]
    model_id = model["model_id"]
    model_id_fs = fs_safe_label(model_id)
    base_model_path = Path(model["path"])
    method_id_fp = "full_precision"
    fp_output_name = f"{model_id_fs}_{method_id_fp}"
    if not cli.skip_eval:
        rows = eval_model_against_jobs(
            model_path=base_model_path,
            syco_repo=syco_repo,
            eval_jobs=eval_jobs,
            data_seeds=data_seeds,
            syco_args=syco_args,
            syco_device=syco_device,
            dataset=syco_dataset,
            model_output_name=fp_output_name,
            log_tag="Syco FP",
            cuda_visible_devices=quant_cuda_visible,
            base_model_name=None,
            model_id=model_id,
            method_id=method_id_fp,
            aggregate_csv=bool(aggregate_csv),
            output_base_map=output_base_map,
            accuracy_script_dir=syco_script_dir,
        )
        accuracy_rows.extend(rows)
        completed_combos.append((model_id_fs, method_id_fp))

    # 4) 量化 + 评估
    assert cli.weight_bits is not None
    try:
        method_combos = [resolve_method_quant_combo(methods[0], cli.weight_bits)]
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    for method_name, method_id, weight_bits in method_combos:
        fake_quant_dir = _quantize_one(
            model_id=model_id,
            model_id_fs=model_id_fs,
            model_path_cfg=model["path"],
            model_type=model["type"],
            method_name=method_name,
            method_id=method_id,
            weight_bits=weight_bits,
            llmc_root=llmc_root,
            cache_root=cache_root,
            calib_root=calib_root,
            eval_root=eval_root,
            calib_auto_download=calib_auto_download,
            eval_auto_download=eval_auto_download,
            base_seed=base_seed,
            nproc=nproc,
            cuda_visible=quant_cuda_visible,
            run_yml_dir=run_yml_dir,
        )
        if fake_quant_dir is None:
            continue
        model_output_name = f"{model_id_fs}_{method_id}"
        if not cli.skip_eval:
            rows = eval_model_against_jobs(
                model_path=fake_quant_dir.resolve(),
                syco_repo=syco_repo,
                eval_jobs=eval_jobs,
                data_seeds=data_seeds,
                syco_args=syco_args,
                syco_device=syco_device,
                dataset=syco_dataset,
                model_output_name=model_output_name,
                log_tag="Syco",
                cuda_visible_devices=quant_cuda_visible,
                base_model_name=model["path"],
                model_id=model_id,
                method_id=method_id,
                aggregate_csv=bool(aggregate_csv),
                output_base_map=output_base_map,
                accuracy_script_dir=syco_script_dir,
            )
            accuracy_rows.extend(rows)
        completed_combos.append((model_id_fs, method_id))

    # 5) aggregate CSV
    if aggregate_csv and accuracy_rows:
        csv_path = resolve_path(aggregate_csv, config_dir) or (script_dir / aggregate_csv)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(accuracy_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(accuracy_rows)
        print(f"Wrote {len(accuracy_rows)} rows to {csv_path}")

    # 6) 绘图阶段（CLI --skip_plot 或 yaml plot_figures=false 时跳过）
    if cli.skip_plot:
        plot_figures = False
    if plot_figures and completed_combos:
        if figure_output_dir:
            fig_dir = str(resolve_path(figure_output_dir, config_dir) or (script_dir / figure_output_dir))
        else:
            fig_dir = str((result_root / "behavioral" / syco_dataset).resolve())
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
        seeds_for_plot = [s for s in data_seeds if s is not None] or [42]
        run_plot_phase(
            syco_repo=syco_repo,
            completed_combos=completed_combos,
            dataset=syco_dataset,
            seeds_for_plot=seeds_for_plot,
            figure_dir=fig_dir,
            eval_authority_advanced=eval_authority_advanced,
            eval_mechanistic=eval_mechanistic,
            eval_sr_correct_only=eval_sr_correct_only,
            eval_jobs=eval_jobs,
            behavioral_output_base=behavioral_base,
            mechanistic_output_base=mechanistic_base,
        )


if __name__ == "__main__":
    main()
