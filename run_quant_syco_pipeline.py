#!/usr/bin/env python3
"""
Multi-model x multi-quantization pipeline: LightCompress (fake_quant) -> LLM-sycophancy.
Uses default public calib datasets and official special params per method (no overrides).
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Default official yml per method (relative to LightCompress root). Calib/special are not overridden.
DEFAULT_METHOD_YML = {
    "GPTQ": "configs/quantization/methods/GPTQ/gptq_w_only.yml",
    "Awq": "configs/quantization/methods/Awq/awq_w_only.yml",
    "RTN": "configs/quantization/methods/RTN/rtn_w_only.yml",
    "QUIK": "configs/quantization/methods/QUIK/quik_w_a.yml",
    "OmniQuant": "configs/quantization/methods/OmniQuant/omniq_w_only.yml",
    "QuaRot": "configs/quantization/methods/QuaRot/quarot_w_a.yml",
    "SmoothQuant": "configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml",
    "HQQ": "configs/quantization/methods/HQQ/hqq_w_only.yml",
    "DGQ": "configs/quantization/methods/DGQ/dgq_w_a.yml",
    "TesseraQ": "configs/quantization/methods/Tesseraq/tesseraq_w_only.yml",
    "NormTweaking": "configs/quantization/methods/NormTweaking/ntweak_w_only.yml",
    "SpQR": "configs/quantization/methods/SpQR/spqr_w_only.yml",
    "LlmInt8": "configs/quantization/methods/LlmInt8/llmint8_w_only.yml",
    "AdaDim": "configs/quantization/methods/AdaDim/adadim_w_a.yml",
    "OsPlus": "configs/quantization/methods/OsPlus/osplus_w_a.yml",
}

# 每个量化方法固定跑 4/6/8 bit（仅配置 method 时生效）
DEFAULT_WEIGHT_BITS = [4, 6, 8]

# Placeholders in official ymls to replace with config roots
CALIB_PATH_PLACEHOLDER = "calib data path"
EVAL_PATH_PLACEHOLDER = "eval data path"

# AWQ 硬编码：忽略前 N 个 block（idx 0..N-1）内所有线性层，保留原精度。
# 走 LightCompress 顶层 ignored_layers 接口（base_blockwise_quantization.set_no_quant_layer）。
# layer_names 必须是相对 block 的子模块短名（Llama/Qwen/Mistral/Gemma 系命名一致）。
AWQ_IGNORE_FIRST_N_BLOCKS = 5
AWQ_IGNORE_LAYER_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Repo layout (relative to script dir)
DIR_LIGHTCOMPRESS = "LightCompress"
DIR_LLM_SYCOPHANCY = "LLM-sycophancy"
DIR_SYCO_SCRIPT = "experiments/behavioral_analysis"

# Timeouts (seconds)
TIMEOUT_SYCO_RUN = 3600 * 24
TIMEOUT_COMPUTE_ACCURACY = 60


def _truthy_inference_debug(v) -> bool:
    """Matches run_syco --debug_inference / RUN_SYCO_DEBUG: enable for true, 1, 2, yes, on."""
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, int):
        return v != 0
    s = str(v).strip().lower()
    return s not in ("", "0", "false", "no", "off")


def _cuda_visible_devices_from_config(val: str | int | None) -> str | None:
    """Return CUDA_VISIBLE_DEVICES string, or None to leave the env unset (all GPUs visible).

    pipeline cuda_device / --gpu may use auto|all|none to mean: do not narrow visibility.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in ("auto", "all", "none", "default"):
        return None
    return s


def build_syco_eval_jobs(
    data_slug: str,
    *,
    eval_mechanistic: bool = True,
    eval_authority_advanced: bool = False,
    eval_behavior_prefix: bool = False,
    behavior_input_filename: str | None = None,
) -> list[dict]:
    """评估任务：plain / opinion_only（run_syco）；可选机理 logit；可选 Academic 学术第一人称（三档）；可选 behavior 前缀。

    input 基名无 _{seed}：run_syco.py 会在指定 data_seed 时插入 _{seed} 再读 pkl。
    behavior：未配置 behavior_input_filename 时使用
    ``lib/behavior/prefix/{data_slug}_behavior_opinion.pkl``（相对 LLM-sycophancy 根）。
    """
    plain_in = f"lib/plain/{data_slug}_plain.pkl"
    opinion_in = f"lib/opinion_only/prefix/{data_slug}_opinion_only.pkl"
    # 与 opinion_only 目录结构对称；run_syco 会在有 data_seed 时插入 _{seed}
    default_behavior_in = f"lib/behavior/prefix/{data_slug}_behavior_opinion.pkl"
    jobs: list[dict] = [
        {
            "script": "run_syco.py",
            "dir": DIR_SYCO_SCRIPT,
            "tag": "plain",
            "for_aggregate": False,
            "kwargs": {"question_type": "plain", "input_filename": plain_in},
        },
        {
            "script": "run_syco.py",
            "dir": DIR_SYCO_SCRIPT,
            "tag": "opinion_only",
            "for_aggregate": True,
            "kwargs": {"question_type": "opinion_only", "input_filename": opinion_in},
        },
    ]
    if eval_mechanistic:
        jobs.extend(
            [
                {
                    "script": "run_syco_logit_cot.py",
                    "dir": "experiments/mechanistic_analysis",
                    "tag": "logit_cot_opinion",
                    "for_aggregate": False,
                    "kwargs": {
                        "question_type": "opinion_only",
                        "inference_mode": "logit_only",
                        "inference_layer": "all",
                        "input_filename": opinion_in,
                    },
                },
                {
                    "script": "run_syco_logit_cot.py",
                    "dir": "experiments/mechanistic_analysis",
                    "tag": "logit_cot_plain",
                    "for_aggregate": False,
                    "kwargs": {
                        "question_type": "plain",
                        "inference_mode": "logit_only",
                        "inference_layer": "all",
                        "input_filename": plain_in,
                    },
                },
            ]
        )
    if eval_authority_advanced:
        # 为了支持 fig2（Beginner / Intermediate / Advanced），三档都执行
        for level in ("beginner", "intermediate", "advanced"):
            level_in = f"lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{level}.pkl"
            jobs.append(
                {
                    "script": "run_syco.py",
                    "dir": DIR_SYCO_SCRIPT,
                    "tag": f"authority_{level}",
                    "for_aggregate": False,
                    "kwargs": {
                        "question_type": "prefix_and_opinion",
                        "prefix_type": "academic",
                        "prefix_subtype": "original",
                        "academic_level": level,
                        "input_filename": level_in,
                    },
                }
            )
    if eval_behavior_prefix:
        b_in = (
            str(behavior_input_filename).strip()
            if behavior_input_filename and str(behavior_input_filename).strip()
            else default_behavior_in
        )
        jobs.append(
            {
                "script": "run_syco.py",
                "dir": DIR_SYCO_SCRIPT,
                "tag": "behavior_prefix",
                "for_aggregate": False,
                "kwargs": {
                    "question_type": "prefix_and_opinion",
                    "prefix_type": "behavior",
                    "prefix_subtype": "original",
                    "input_filename": b_in,
                },
            },
        )
    return jobs


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(value: str | None, config_dir: Path) -> Path | None:
    if not value or value.startswith("/"):
        return Path(value) if value else None
    return (config_dir / value).resolve()


def fs_safe_label(s: str) -> str:
    """HF 的 org/model 等形式含 '/'，不能直接用作单层目录名或文件名片段。"""
    return s.replace("\\", "_").replace("/", "_")


def parse_weight_bits_from_method_id(method_id: str) -> int | None:
    """从 method_id 解析权重量化比特数，如 rtn_w2 -> 2, gptq_w4 -> 4。若无 _wN 则返回 None。"""
    m = re.search(r"_w(\d+)$", method_id.strip())
    return int(m.group(1)) if m else None


def expand_methods_to_bits(methods: list) -> list[tuple[str, str, int]]:
    """将 methods 配置展开为 (method_name, method_id, weight_bits) 列表。仅配置 method 时固定跑 4/6/8 bit。"""
    result = []
    for m in methods:
        method_name = m.get("method")
        method_id = m.get("method_id")
        if method_name not in DEFAULT_METHOD_YML:
            continue
        if method_id is not None:
            bits = parse_weight_bits_from_method_id(method_id)
            result.append((method_name, method_id, bits if bits is not None else 8))
        else:
            for b in DEFAULT_WEIGHT_BITS:
                result.append((method_name, f"{method_name.lower()}_w{b}", b))
    return result


def build_run_yml(
    template_path: Path,
    model_path: str,
    model_type: str,
    save_path: Path,
    calib_root: Path,
    eval_root: Path,
    base_seed: int,
    calib_auto_download: bool = False,
    eval_auto_download: bool = False,
    weight_bits: int | None = None,
    ignored_layers: dict | None = None,
) -> dict:
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(CALIB_PATH_PLACEHOLDER, str(calib_root))
    content = content.replace(EVAL_PATH_PLACEHOLDER, str(eval_root))
    data = yaml.safe_load(content)
    if data is None:
        data = {}
    if "model" not in data:
        data["model"] = {}
    data["model"]["path"] = model_path
    data["model"]["type"] = model_type
    if "save" not in data:
        data["save"] = {}
    data["save"]["save_path"] = str(save_path)
    data["save"]["save_fake"] = True
    if "base" not in data:
        data["base"] = {}
    data["base"]["seed"] = base_seed
    if "calib" in data:
        data["calib"]["download"] = calib_auto_download
    if "eval" in data:
        data["eval"]["download"] = eval_auto_download
    if weight_bits is not None and "quant" in data and "weight" in data["quant"]:
        data["quant"]["weight"]["bit"] = weight_bits
    if ignored_layers:
        data["ignored_layers"] = ignored_layers
    return data


def run_lightcompress(
    llmc_root: Path,
    run_yml_path: Path,
    task_id: str,
    nproc_per_node: int,
    cuda_devices: str | None = None,
) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(llmc_root)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if cuda_devices is not None:
        # 只影响 LightCompress 这一步的可见 GPU
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    port = 29500 + (hash(task_id) % 30000)
    if port < 10000:
        port += 20000
    endpoint = f"127.0.0.1:{port}"
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", "1",
        "--nproc_per_node", str(nproc_per_node),
        "--rdzv_id", task_id,
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", endpoint,
        str(llmc_root / "llmc" / "__main__.py"),
        "--config", str(run_yml_path),
        "--task_id", task_id,
    ]
    try:
        ret = subprocess.run(cmd, env=env, cwd=str(llmc_root))
        return ret.returncode == 0
    except Exception as e:
        print(f"LightCompress run failed: {e}", file=sys.stderr)
        return False


def _resolve_input_path(filename: str, repo_root: Path) -> str:
    if not filename or os.path.isabs(filename):
        return filename
    return str((repo_root / filename).resolve())


def _resolve_seeded_input_path(filename: str, repo_root: Path, data_seed: int | None) -> Path:
    """按 run_syco 的规则把 data_seed 后缀插入到输入 pkl 文件名。"""
    p = Path(_resolve_input_path(filename, repo_root))
    if data_seed is None:
        return p
    return p.with_name(f"{p.stem}_{data_seed}{p.suffix}")


def _split_method_and_seed_from_stem(stem: str, model_id_fs: str) -> tuple[str | None, int | None]:
    """从 {model_id_fs}_{method_id}[_seed].pkl 的 stem 解析 method_id 与 seed。"""
    prefix = f"{model_id_fs}_"
    if not stem.startswith(prefix):
        return None, None
    body = stem[len(prefix) :]
    m = re.match(r"^(.*)_(\d+)$", body)
    if m:
        return m.group(1), int(m.group(2))
    return body, None


def _scan_existing_methods_for_plot(
    ba_dir: Path,
    dataset: str,
    model_id_fs: str,
    seeds: list[int] | None,
) -> list[str]:
    """扫描 output/{dataset}/plain 下已有 pkl，返回可用于绘图的 method_id 列表。"""
    plain_dir = ba_dir / "output" / dataset / "plain"
    if not plain_dir.exists():
        return []
    methods = set()
    for p in plain_dir.glob(f"{model_id_fs}_*.pkl"):
        method_id, seed = _split_method_and_seed_from_stem(p.stem, model_id_fs)
        if not method_id:
            continue
        if seeds is not None and seed not in seeds:
            continue
        # 至少要求 opinion_only 同名文件存在，避免 fig1 直接失败
        opinion_name = p.name
        op_path = ba_dir / "output" / dataset / "opinion_only" / opinion_name
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
) -> int:
    """只基于已有 pkl 批量绘图（不跑量化/评测）。返回成功绘图数量。"""
    ba_dir = syco_repo / DIR_SYCO_SCRIPT
    methods = _scan_existing_methods_for_plot(ba_dir, dataset, model_id_fs, seeds_for_plot if seeds_for_plot else None)
    if not methods:
        print(
            f"[PlotOnly] no eligible methods found in output/{dataset}/plain for model_id_fs={model_id_fs}",
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
            "--which",
            "fig1",
            "--output_base",
            "output",
            "--dataset_subdir",
            dataset,
            "--figure_dir",
            figure_dir,
            "--model_type",
            plot_output_name,
            "--output_suffix",
            out_suffix,
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


def _ensure_syco_data_for_seed(syco_repo: Path, seed: int, syco_cfg: dict) -> bool:
    """若该 seed 对应的 plain / opinion_only pkl 不存在则下载 raw 并运行 build_lib_from_raw。"""
    dataset = syco_cfg.get("dataset", "mmlu")
    data_slug = syco_cfg.get("data_slug") or dataset
    raw_rel = syco_cfg.get("raw_file", "raw_data/mmlu_raw.pkl")
    download_name = syco_cfg.get("download_script", "download_mmlu.py")
    extra = syco_cfg.get("build_lib_extra_args") or []
    if not isinstance(extra, list):
        extra = list(extra) if extra else []

    plain_pkl = syco_repo / "lib" / "plain" / f"{data_slug}_plain_{seed}.pkl"
    opinion_pkl = syco_repo / "lib" / "opinion_only" / "prefix" / f"{data_slug}_opinion_only_{seed}.pkl"
    if plain_pkl.exists() and opinion_pkl.exists():
        return True

    raw_pkl = (syco_repo / raw_rel).resolve() if not os.path.isabs(raw_rel) else Path(raw_rel)
    dg_dir = syco_repo / "experiments" / "data_generation"
    download_script = dg_dir / download_name
    if not raw_pkl.exists():
        if not download_script.exists():
            print(f"缺少 raw 数据且未找到 {download_script}", file=sys.stderr)
            return False
        raw_pkl.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在下载原始数据到 {raw_pkl}（{download_name}）...")
        try:
            r = subprocess.run(
                [sys.executable, str(download_script), "--output", str(raw_pkl)],
                cwd=str(dg_dir),
                timeout=600,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"{download_name} 失败: {r.stderr or r.stdout}", file=sys.stderr)
                return False
        except subprocess.TimeoutExpired:
            print(f"{download_name} 超时", file=sys.stderr)
            return False
        if not raw_pkl.exists():
            print(f"下载后仍不存在: {raw_pkl}", file=sys.stderr)
            return False

    build_script = dg_dir / "build_lib_from_raw.py"
    if not build_script.exists():
        print(f"未找到 {build_script}", file=sys.stderr)
        return False
    print(f"正在为 seed={seed} 生成 lib plain/opinion_only（data_slug={data_slug}）...")
    try:
        cmd = [
            sys.executable,
            str(build_script),
            "--seed",
            str(seed),
            "--raw_file",
            raw_rel,
        ]
        cmd.extend(str(x) for x in extra)
        r = subprocess.run(
            cmd,
            cwd=str(syco_repo),
            timeout=600,
            capture_output=True,
            text=True,
        )
        if not plain_pkl.exists() or not opinion_pkl.exists():
            if r.returncode != 0 and r.stderr:
                print(r.stderr, file=sys.stderr)
            print(f"生成后仍缺少 pkl: plain={plain_pkl.exists()}, opinion={opinion_pkl.exists()}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("build_lib_from_raw 超时", file=sys.stderr)
        return False


def _expected_syco_pkl_path(script_dir: Path, eval_job: dict, model_path: Path, dataset: str, data_seed: int | None = None, model_output_name: str | None = None) -> Path:
    """与 run_syco.py / run_syco_logit_cot.py 的保存路径规则一致，用于不 capture_output 时推断输出路径。"""
    if model_output_name:
        basename = model_output_name
    else:
        basename = model_path.name if model_path.is_dir() else model_path.stem
        basename = basename.replace(".", "_")
    kwargs = eval_job.get("kwargs", {})
    question_type = kwargs.get("question_type", "plain")
    prefix_type = kwargs.get("prefix_type", "")
    prefix_subtype = kwargs.get("prefix_subtype", "")
    academic_level = kwargs.get("academic_level", "")
    seed_suffix = f"_{data_seed}" if data_seed is not None else ""
    if eval_job["script"] == "run_syco_logit_cot.py":
        out_base = "output_inference"
        inference_mode = kwargs.get("inference_mode", "logit_only")
        inference_layer = kwargs.get("inference_layer", "all")
        mode_str = "cot" if inference_mode == "logit_and_cot" else "logit"
        parts = [out_base, dataset, question_type]
        if prefix_type:
            parts.extend([prefix_type, prefix_subtype, academic_level])
        rel_dir = os.path.join(*[p for p in parts if p])
        fname = f"{basename}_{mode_str}_{inference_layer}{seed_suffix}.pkl"
    else:
        out_base = "output"
        parts = [out_base, dataset, question_type]
        if prefix_type:
            parts.extend([prefix_type, prefix_subtype, academic_level])
        rel_dir = os.path.join(*[p for p in parts if p])
        fname = f"{basename}{seed_suffix}.pkl"
    return (script_dir / rel_dir / fname).resolve()


def run_syco_eval(
    model_path: Path,
    syco_repo_root: Path,
    eval_job: dict,
    device: str = "auto",
    dataset: str = "mmlu",
    full_question_column: str = "full_question",
    max_retries: int = 3,
    base_model_name: str | None = None,
    data_seed: int | None = None,
    model_output_name: str | None = None,
    cuda_visible_devices: str | None = None,
) -> str | None:
    """Run one syco eval (run_syco.py or run_syco_logit_cot.py). Return output pkl path if parseable."""
    script_name = eval_job["script"]
    script_dir = syco_repo_root / eval_job["dir"]
    kwargs = dict(eval_job["kwargs"])
    input_path = _resolve_input_path(kwargs.pop("input_filename", ""), syco_repo_root)
    cmd = [
        sys.executable, script_name,
        "--model_name", str(model_path),
        "--dataset", dataset,
        "--input_filename", input_path,
        "--full_question_column", full_question_column,
        "--max_retries", str(max_retries),
        "--device", device,
    ]
    if data_seed is not None:
        cmd.extend(["--data_seed", str(data_seed)])
    if model_output_name:
        cmd.extend(["--model_output_name", model_output_name])
    if base_model_name is not None and "fake_quant_model" in str(model_path):
        cmd.extend(["--base_model_name", base_model_name])
    for k, v in kwargs.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            continue
        cmd.extend([f"--{k}", str(v)])
    expected_pkl = _expected_syco_pkl_path(script_dir, eval_job, model_path, dataset, data_seed=data_seed, model_output_name=model_output_name)
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            env=env,
            stdout=None,
            stderr=None,
            timeout=TIMEOUT_SYCO_RUN,
        )
        if result.returncode != 0:
            return None
        if expected_pkl.exists():
            return str(expected_pkl)
        return None
    except subprocess.TimeoutExpired:
        print(f"{script_name} timed out after {TIMEOUT_SYCO_RUN}s (eval {eval_job.get('tag', '')}), skipping.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"{script_name} failed: {e}", file=sys.stderr)
        return None


def run_syco(
    syco_script_dir: Path,
    syco_repo_root: Path,
    model_path: Path,
    syco_args: dict,
) -> str | None:
    """Run run_syco.py with a single args dict; return path to output pkl if found. (Legacy single-eval.)"""
    job = {
        "script": "run_syco.py",
        "dir": DIR_SYCO_SCRIPT,
        "kwargs": {
            "question_type": syco_args.get("question_type", "plain"),
            "prefix_type": syco_args.get("prefix_type", ""),
            "prefix_subtype": syco_args.get("prefix_subtype", "original"),
            "academic_level": syco_args.get("academic_level", ""),
            "input_filename": syco_args.get("input_filename", ""),
        },
    }
    return run_syco_eval(
        model_path,
        syco_repo_root,
        job,
        device=syco_args.get("device", "auto"),
        dataset=syco_args.get("dataset", "mmlu"),
        full_question_column=syco_args.get("full_question_column", "full_question"),
        max_retries=syco_args.get("max_retries", 3),
    )


def run_compute_accuracy(pkl_path: str, script_dir: Path) -> dict | None:
    """Run compute_accuracy.py on pkl; return parsed metrics if opinion_only."""
    cmd = [sys.executable, "compute_accuracy.py", pkl_path]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_COMPUTE_ACCURACY,
        )
        out = (result.stdout or "") + (result.stderr or "")
        metrics = {}
        # opinion_only: 正确 / 顺从 / 其他
        m = re.search(r"正确 \(选对\):\s+([\d.]+)", out)
        if m:
            metrics["correct_pct"] = float(m.group(1))
        m = re.search(r"顺从错误观点:\s+([\d.]+)", out)
        if m:
            metrics["sycophantic_pct"] = float(m.group(1))
        m = re.search(r"其他:\s+([\d.]+)", out)
        if m:
            metrics["other_pct"] = float(m.group(1))
        if not metrics:
            m = re.search(r"准确率:\s+([\d.]+)", out)
            if m:
                metrics["accuracy"] = float(m.group(1))
        return metrics if metrics else None
    except Exception as e:
        print(f"compute_accuracy failed: {e}", file=sys.stderr)
        return None


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "pipeline_config.yaml"

    # 解析命令行：第一个非选项参数作为 config，其余参数用于覆盖/附加配置
    extra_args = []
    if len(sys.argv) > 1:
        # 找到第一个不是以 - 开头的参数，认为是 config 路径
        cfg_arg_idx = None
        for i, a in enumerate(sys.argv[1:], start=1):
            if not a.startswith("-"):
                cfg_arg_idx = i
                break
        if cfg_arg_idx is not None:
            config_path = Path(sys.argv[cfg_arg_idx]).resolve()
            extra_args = sys.argv[cfg_arg_idx + 1 :]
        else:
            extra_args = sys.argv[1:]

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    config_dir = config_path.parent

    save_root = resolve_path(cfg.get("save_root"), config_dir)
    calib_root = resolve_path(cfg.get("calib_root"), config_dir)
    eval_root = resolve_path(cfg.get("eval_root"), config_dir)
    if not save_root:
        print("save_root must be set in config.", file=sys.stderr)
        sys.exit(1)
    if calib_root is None:
        calib_root = config_dir
    if eval_root is None:
        eval_root = config_dir
    calib_auto_download = cfg.get("calib_auto_download", False)
    eval_auto_download = cfg.get("eval_auto_download", False)

    # 将 Hugging Face 缓存与临时文件重定向到 save_root 下，避免占用默认家目录磁盘
    if save_root is not None:
        hf_cache_dir = save_root / "_hf_cache"
        tmp_dir = save_root / "_tmp"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("TMPDIR", str(tmp_dir))

    llmc_root = resolve_path(cfg.get("llmc_dir"), config_dir) or (script_dir / DIR_LIGHTCOMPRESS)
    syco_repo = resolve_path(cfg.get("syco_repo_dir"), config_dir) or (script_dir / DIR_LLM_SYCOPHANCY)
    syco_script_dir = syco_repo / DIR_SYCO_SCRIPT
    if not llmc_root.exists():
        print(f"LightCompress not found at {llmc_root}", file=sys.stderr)
        print("Set config 'llmc_dir' to the LightCompress repo path if it is elsewhere.", file=sys.stderr)
        sys.exit(1)
    if not syco_script_dir.exists():
        print(f"LLM-sycophancy experiments not found at {syco_script_dir}", file=sys.stderr)
        print("Ensure LLM-sycophancy is under the same dir as this script, or set config 'syco_repo_dir'.", file=sys.stderr)
        sys.exit(1)

    # plot-only 模式：扫描已有 pkl 并批量画图，不跑量化/评测
    if "--plot_scan_existing" in extra_args:
        def _get_arg(name: str) -> str | None:
            for i, a in enumerate(extra_args):
                if a == name and i + 1 < len(extra_args):
                    return extra_args[i + 1]
                if a.startswith(name + "="):
                    return a.split("=", 1)[1]
            return None

        scan_dataset = _get_arg("--plot_scan_dataset") or str((cfg.get("syco", {}) or {}).get("dataset", "mmlu"))
        scan_model_id = _get_arg("--plot_scan_model_id")
        if not scan_model_id:
            print("plot-only requires --plot_scan_model_id <model_id>", file=sys.stderr)
            sys.exit(1)
        scan_model_id_fs = fs_safe_label(scan_model_id)
        seeds_str = _get_arg("--plot_scan_seeds")
        if seeds_str:
            scan_seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
        else:
            _data_seed_raw = ((cfg.get("syco", {}) or {}).get("data_seed", 42))
            if isinstance(_data_seed_raw, list):
                scan_seeds = [int(x) for x in _data_seed_raw if x is not None]
            elif _data_seed_raw is None:
                scan_seeds = [42]
            else:
                scan_seeds = [int(_data_seed_raw)]
        scan_figure_dir = _get_arg("--plot_scan_figure_dir") or "figure"
        correct_only_flag = _get_arg("--plot_scan_correct_only")
        if correct_only_flag is None:
            scan_correct_only = bool(((cfg.get("syco", {}) or {}).get("eval_sr_correct_only", False)))
        else:
            scan_correct_only = str(correct_only_flag).strip().lower() in ("1", "true", "yes", "on")
        run_plot_from_existing_pkls(
            syco_repo=syco_repo,
            dataset=scan_dataset,
            model_id_fs=scan_model_id_fs,
            seeds_for_plot=scan_seeds,
            figure_dir=scan_figure_dir,
            correct_only_sr=scan_correct_only,
        )
        return

    models = cfg.get("models") or []
    methods = cfg.get("methods") or []
    if not methods:
        print(
            "pipeline_config.yaml 中 methods 未配置任何量化方法（键为空或列表项均被注释），"
            "将跳过 LightCompress 量化阶段。",
            file=sys.stderr,
        )
    syco_args = cfg.get("syco", {})
    _syco_dataset = syco_args.get("dataset", "mmlu")
    _data_slug = syco_args.get("data_slug") or _syco_dataset
    eval_mechanistic = bool(syco_args.get("eval_mechanistic", True))
    eval_authority_advanced = bool(syco_args.get("eval_authority_advanced", False))
    eval_behavior_prefix = bool(syco_args.get("eval_behavior_prefix", False))
    behavior_input_filename = syco_args.get("behavior_input_filename")
    eval_sr_correct_only = bool(syco_args.get("eval_sr_correct_only", False))
    eval_jobs = build_syco_eval_jobs(
        _data_slug,
        eval_mechanistic=eval_mechanistic,
        eval_authority_advanced=eval_authority_advanced,
        eval_behavior_prefix=eval_behavior_prefix,
        behavior_input_filename=behavior_input_filename,
    )
    nproc = cfg.get("nproc_per_node", 1)
    # 量化与 syco 共用 GPU：优先命令行 --gpu / --gpu=，其次配置 cuda_device（或兼容旧版 quant_cuda_devices）
    cuda_device_cfg = cfg.get("cuda_device") or cfg.get("quant_cuda_devices")
    quant_cuda_devices_cli = None
    i = 0
    while i < len(extra_args):
        a = extra_args[i]
        if a.startswith("--gpu="):
            quant_cuda_devices_cli = a.split("=", 1)[1]
            break
        if a == "--gpu" and i + 1 < len(extra_args):
            quant_cuda_devices_cli = extra_args[i + 1]
            break
        i += 1
    quant_cuda_devices = quant_cuda_devices_cli or cuda_device_cfg
    quant_cuda_visible = _cuda_visible_devices_from_config(quant_cuda_devices)
    # syco --device：已设 CUDA_VISIBLE_DEVICES 时子进程内仅见首张卡，逻辑编号恒为 cuda:0（勿用物理卡号）
    if quant_cuda_visible is not None:
        syco_device = "cuda:0"
    else:
        syco_device = syco_args.get("device", "cuda:0")
    base_seed = 42  # 量化固定 42
    aggregate_csv = cfg.get("aggregate_accuracy_csv")
    plot_figures = cfg.get("plot_figures", True)
    figure_output_dir = cfg.get("figure_output_dir")  # optional; else each script uses its default

    # 评估用 syco.data_seed：单值或列表 [42, 123, 456]
    _data_seed_raw = syco_args.get("data_seed", 42)
    if _data_seed_raw is None:
        data_seeds = [None]
    elif isinstance(_data_seed_raw, list):
        data_seeds = [s if s is None else int(s) for s in _data_seed_raw]
    else:
        data_seeds = [int(_data_seed_raw)]

    # 对每个用到的随机种子，若无对应 pkl 则自动生成
    for s in data_seeds:
        if s is not None:
            if not _ensure_syco_data_for_seed(syco_repo, s, syco_args):
                print(f"无法为 data_seed={s} 准备数据，退出", file=sys.stderr)
                sys.exit(1)
            if eval_authority_advanced:
                fp_dir = syco_repo / "lib" / "pov" / "prefix" / "first_pov"
                required_levels = ("beginner", "intermediate", "advanced")
                missing = []
                for level in required_levels:
                    p = fp_dir / f"{_data_slug}_academic_opinion_{level}_{s}.pkl"
                    if not p.exists():
                        missing.append((level, p))
                if missing:
                    hint = ""
                    if fp_dir.is_dir():
                        all_glob = sorted(fp_dir.glob(f"{_data_slug}_academic_opinion_*_{s}.pkl"))
                        if all_glob:
                            names = ", ".join(p.name for p in all_glob[:12])
                            more = " …" if len(all_glob) > 12 else ""
                            hint = (
                                f"\n同目录下已找到（{_data_slug}, seed={s}）: {names}{more}"
                                f"\n当前要求三档文件: beginner / intermediate / advanced。"
                            )
                    miss_text = "\n".join([f"- {lvl}: {path}" for lvl, path in missing])
                    print(
                        "已开启 eval_authority_advanced，但缺少 Academic 三档输入文件：\n"
                        f"{miss_text}\n"
                        f"请先对 seed={s} 运行 generate_prefixes.py 与 build_lib_from_raw.py（与 build_lib_extra_args 一致）。"
                        f"{hint}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    run_dir = save_root / "_run_configs"
    run_dir.mkdir(parents=True, exist_ok=True)

    accuracy_rows = []
    completed_combos = []  # (model_id, method_id) for plot phase

    # 先对每个模型在 full precision 下做一次 syco 评估（若已存在则跳过）
    for model in models:
        model_id = model["model_id"]
        model_id_fs = fs_safe_label(model_id)
        model_path_str = model["path"]
        base_model_path = Path(model_path_str)

        device = syco_device
        dataset = _syco_dataset
        full_question_column = syco_args.get("full_question_column", "full_question")
        max_retries = syco_args.get("max_retries", 3)
        method_id = "full_precision"
        model_output_name = f"{model_id_fs}_{method_id}"

        for data_seed in data_seeds:
            pkl_for_aggregate = None
            # 必须逐个 eval_job 检查/跳过：若仅因 opinion_only 已存在就短路整个循环，
            # 会导致 beginner/intermediate/advanced 等任务永远不跑。
            for eval_job in eval_jobs:
                ej = dict(eval_job)
                if ej.get("script") == "run_syco.py":
                    kwd = dict(ej.get("kwargs", {}))
                    amnt = syco_args.get("answer_max_new_tokens")
                    if amnt is not None:
                        kwd["answer_max_new_tokens"] = int(amnt)
                    if _truthy_inference_debug(syco_args.get("debug_inference")):
                        kwd["debug_inference"] = True
                    ej["kwargs"] = kwd
                if ej.get("script") == "run_syco_logit_cot.py":
                    kwd = dict(ej.get("kwargs", {}))
                    if syco_args.get("save_option_hs"):
                        kwd["save_option_hs"] = True
                    shd = syco_args.get("stream_hs_dir")
                    if shd:
                        kwd["stream_hs_dir"] = str(shd)
                    ej["kwargs"] = kwd
                # 若对应评估输出 pkl 已存在，则跳过该评估
                expected_eval_pkl = _expected_syco_pkl_path(
                    syco_repo / ej["dir"],
                    ej,
                    base_model_path,
                    dataset,
                    data_seed=data_seed,
                    model_output_name=model_output_name,
                )
                if expected_eval_pkl.exists():
                    print(
                        f"[Syco FP] {model_id} x {method_id} (seed={data_seed}) -> "
                        f"skip {eval_job['tag']} (already exists: {expected_eval_pkl.name})"
                    )
                    pkl_path = str(expected_eval_pkl)
                    if eval_job.get("for_aggregate"):
                        pkl_for_aggregate = pkl_path
                    continue
                input_filename = str(ej.get("kwargs", {}).get("input_filename", "")).strip()
                if input_filename:
                    resolved_input = _resolve_seeded_input_path(input_filename, syco_repo, data_seed)
                    if not resolved_input.exists():
                        print(
                            f"[Syco FP] {model_id} x {method_id} (seed={data_seed}) -> "
                            f"skip {eval_job['tag']} (missing input: {resolved_input})",
                            file=sys.stderr,
                        )
                        continue
                print(f"[Syco FP] {model_id} x {method_id} (seed={data_seed}) -> {eval_job['tag']}")
                pkl_path = run_syco_eval(
                    base_model_path,
                    syco_repo,
                    ej,
                    device=device,
                    dataset=dataset,
                    full_question_column=full_question_column,
                    max_retries=max_retries,
                    base_model_name=None,
                    data_seed=data_seed,
                    model_output_name=model_output_name,
                    cuda_visible_devices=quant_cuda_visible,
                )
                if eval_job.get("for_aggregate") and pkl_path:
                    pkl_for_aggregate = pkl_path
            if aggregate_csv and pkl_for_aggregate:
                metrics = run_compute_accuracy(pkl_for_aggregate, syco_script_dir)
                if metrics:
                    row = {
                        "model_id": model_id,
                        "method_id": method_id,
                        "data_seed": data_seed,
                        "pkl": pkl_for_aggregate,
                        **metrics,
                    }
                    accuracy_rows.append(row)
        # full precision 也加入后续绘图阶段（与 pkl 命名一致，须用 fs_safe）
        completed_combos.append((model_id_fs, method_id))

    method_combos = expand_methods_to_bits(methods)
    for model in models:
        model_id = model["model_id"]
        model_id_fs = fs_safe_label(model_id)
        model_path = model["path"]
        model_type = model["type"]
        for method_name, method_id, weight_bits in method_combos:
            rel_yml = DEFAULT_METHOD_YML[method_name]
            template_path = llmc_root / rel_yml
            if not template_path.exists():
                print(f"Template yml not found: {template_path}", file=sys.stderr)
                continue

            save_path = save_root / model_id_fs / method_id
            fake_quant_dir = save_path / "fake_quant_model"

            # 只有当目录存在且非空时才认为量化已完成；空目录会删除后重新跑量化。跳过量化后仍会跑评估。
            need_run_quant = False
            if fake_quant_dir.exists():
                if any(fake_quant_dir.iterdir()):
                    print(f"[Skip Quant] {model_id} x {method_id}: fake_quant_model already exists and is non-empty at {fake_quant_dir}")
                else:
                    print(f"[Quant] {model_id} x {method_id}: empty fake_quant_model dir found, removing and re-running quant.")
                    try:
                        fake_quant_dir.rmdir()
                        need_run_quant = True
                    except OSError as e:
                        print(f"Failed to remove empty fake_quant_model dir {fake_quant_dir}: {e}", file=sys.stderr)
                        continue
            else:
                need_run_quant = True

            if need_run_quant:
                task_id = f"{model_id_fs}_{method_id}_{int(time.time())}"
                ignored_layers_cfg = None
                if method_name == "Awq":
                    ignored_layers_cfg = {
                        "block_ids": list(range(AWQ_IGNORE_FIRST_N_BLOCKS)),
                        "layer_names": list(AWQ_IGNORE_LAYER_NAMES),
                    }
                run_yml = build_run_yml(
                    template_path,
                    model_path,
                    model_type,
                    save_path,
                    calib_root,
                    eval_root,
                    base_seed,
                    calib_auto_download=calib_auto_download,
                    eval_auto_download=eval_auto_download,
                    weight_bits=weight_bits,
                    ignored_layers=ignored_layers_cfg,
                )
                run_yml_path = run_dir / f"{task_id}.yaml"
                with open(run_yml_path, "w", encoding="utf-8") as f:
                    yaml.dump(run_yml, f, allow_unicode=True, default_flow_style=False)
                if quant_cuda_visible is not None:
                    print(f"[Quant] {model_id} x {method_id} (task_id={task_id}, CUDA_VISIBLE_DEVICES={quant_cuda_visible})")
                else:
                    print(f"[Quant] {model_id} x {method_id} (task_id={task_id}, CUDA_VISIBLE_DEVICES unset)")
                print(f"  -> weight_bits={weight_bits}")
                if ignored_layers_cfg is not None:
                    print(
                        f"  -> AWQ ignored_layers: first {AWQ_IGNORE_FIRST_N_BLOCKS} blocks "
                        f"({ignored_layers_cfg['block_ids']}) x {ignored_layers_cfg['layer_names']}"
                    )
                if not run_lightcompress(llmc_root, run_yml_path, task_id, nproc, cuda_devices=quant_cuda_visible):
                    print(f"LightCompress failed for {model_id} x {method_id}", file=sys.stderr)
                    continue
                if not fake_quant_dir.exists():
                    print(f"fake_quant_model not found at {fake_quant_dir}", file=sys.stderr)
                    continue

            device = syco_device
            dataset = _syco_dataset
            full_question_column = syco_args.get("full_question_column", "full_question")
            max_retries = syco_args.get("max_retries", 3)
            model_output_name = f"{model_id_fs}_{method_id}"
            for data_seed in data_seeds:
                pkl_for_aggregate = None
                for eval_job in eval_jobs:
                    ej = dict(eval_job)
                    if ej.get("script") == "run_syco.py":
                        kwd = dict(ej.get("kwargs", {}))
                        amnt = syco_args.get("answer_max_new_tokens")
                        if amnt is not None:
                            kwd["answer_max_new_tokens"] = int(amnt)
                        if _truthy_inference_debug(syco_args.get("debug_inference")):
                            kwd["debug_inference"] = True
                        ej["kwargs"] = kwd
                    if ej.get("script") == "run_syco_logit_cot.py":
                        kwd = dict(ej.get("kwargs", {}))
                        if syco_args.get("save_option_hs"):
                            kwd["save_option_hs"] = True
                        shd = syco_args.get("stream_hs_dir")
                        if shd:
                            kwd["stream_hs_dir"] = str(shd)
                        ej["kwargs"] = kwd
                    # 若对应评估输出 pkl 已存在，则跳过该评估
                    expected_eval_pkl = _expected_syco_pkl_path(
                        syco_repo / ej["dir"],
                        ej,
                        fake_quant_dir.resolve(),
                        dataset,
                        data_seed=data_seed,
                        model_output_name=model_output_name,
                    )
                    if expected_eval_pkl.exists():
                        print(
                            f"[Syco] {model_id} x {method_id} (seed={data_seed}) -> "
                            f"skip {eval_job['tag']} (already exists: {expected_eval_pkl.name})"
                        )
                        pkl_path = str(expected_eval_pkl)
                        if eval_job.get("for_aggregate"):
                            pkl_for_aggregate = pkl_path
                        continue
                    input_filename = str(ej.get("kwargs", {}).get("input_filename", "")).strip()
                    if input_filename:
                        resolved_input = _resolve_seeded_input_path(input_filename, syco_repo, data_seed)
                        if not resolved_input.exists():
                            print(
                                f"[Syco] {model_id} x {method_id} (seed={data_seed}) -> "
                                f"skip {eval_job['tag']} (missing input: {resolved_input})",
                                file=sys.stderr,
                            )
                            continue
                    print(f"[Syco] {model_id} x {method_id} (seed={data_seed}) -> {eval_job['tag']}")
                    pkl_path = run_syco_eval(
                        fake_quant_dir.resolve(),
                        syco_repo,
                        ej,
                        device=device,
                        dataset=dataset,
                        full_question_column=full_question_column,
                        max_retries=max_retries,
                        base_model_name=model_path,
                        data_seed=data_seed,
                        model_output_name=model_output_name,
                        cuda_visible_devices=quant_cuda_visible,
                    )
                    if eval_job.get("for_aggregate") and pkl_path:
                        pkl_for_aggregate = pkl_path
                if aggregate_csv and pkl_for_aggregate:
                    metrics = run_compute_accuracy(pkl_for_aggregate, syco_script_dir)
                    if metrics:
                        row = {"model_id": model_id, "method_id": method_id, "data_seed": data_seed, "pkl": pkl_for_aggregate, **metrics}
                        accuracy_rows.append(row)
            completed_combos.append((model_id_fs, method_id))

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

    # Plot phase: 每个 (model, method) 画一张图，多 seed 取平均
    if plot_figures and completed_combos:
        fig_dir = figure_output_dir if figure_output_dir else "figure"
        ba_dir = syco_repo / DIR_SYCO_SCRIPT
        mech_dir = syco_repo / "experiments" / "mechanistic_analysis"
        logit_plain_job = next((j for j in eval_jobs if j.get("tag") == "logit_cot_plain"), None)
        logit_opinion_job = next((j for j in eval_jobs if j.get("tag") == "logit_cot_opinion"), None)
        dummy_path = Path("dummy")
        seeds_for_plot = [s for s in data_seeds if s is not None]
        if not seeds_for_plot:
            seeds_for_plot = [42]

        for plot_model_id, plot_method_id in completed_combos:
            plot_output_name = f"{plot_model_id}_{plot_method_id}"
            # 输出图文件名包含数据集，避免多数据集跑同一 figure 目录时互相覆盖
            plot_name_with_dataset = f"{_syco_dataset}_{plot_output_name}"
            if eval_sr_correct_only:
                plot_name_with_dataset = f"{plot_name_with_dataset}_correct_only"

            # 1) plot_figure2.py (fig1): Plain vs Opinion-Only，多 seed 取平均，一张图
            plot_fig1_cmd = [
                sys.executable, "plot_figure2.py",
                "--which", "fig1",
                "--output_base", "output",
                "--dataset_subdir", _syco_dataset,
                "--figure_dir", fig_dir,
                "--model_type", plot_output_name,
                "--output_suffix", plot_name_with_dataset,
                "--data_seeds", *[str(s) for s in seeds_for_plot],
            ]
            if eval_sr_correct_only:
                plot_fig1_cmd.extend(
                    [
                        "--correct_only_sr",
                        "--baseline_model_type",
                        f"{plot_model_id}_full_precision",
                    ]
                )
            print(f"[Plot] Running plot_figure2.py(fig1) for {plot_name_with_dataset} (avg over seeds {seeds_for_plot}) ...")
            subprocess.run(plot_fig1_cmd, cwd=str(ba_dir), timeout=300)

            # 1.1) plot_figure2.py (fig2): First-pov Academic（Beginner / Intermediate / Advanced）
            # 仅在启用 eval_authority_advanced 时触发，避免缺少目录数据导致无意义报错。
            if eval_authority_advanced:
                plot_fig2_adv_cmd = [
                    sys.executable, "plot_figure2.py",
                    "--which", "fig2",
                    "--output_base", "output",
                    "--dataset_subdir", _syco_dataset,
                    "--figure_dir", fig_dir,
                    "--model_type", plot_output_name,
                    "--output_suffix", plot_name_with_dataset,
                    "--data_seeds", *[str(s) for s in seeds_for_plot],
                ]
                print(f"[Plot] Running plot_figure2.py(fig2 advanced) for {plot_name_with_dataset} (avg over seeds {seeds_for_plot}) ...")
                subprocess.run(plot_fig2_adv_cmd, cwd=str(ba_dir), timeout=300)

            # 2) compute_decision_score.py：传入所有 seed 的 pkl，脚本内按 layer 取平均（机理分支）
            if eval_mechanistic and logit_plain_job and logit_opinion_job:
                plain_paths = []
                opinion_paths = []
                for s in seeds_for_plot:
                    pp = _expected_syco_pkl_path(mech_dir, logit_plain_job, dummy_path, _syco_dataset, data_seed=s, model_output_name=plot_output_name)
                    op = _expected_syco_pkl_path(mech_dir, logit_opinion_job, dummy_path, _syco_dataset, data_seed=s, model_output_name=plot_output_name)
                    if pp.exists():
                        plain_paths.append(str(pp))
                    if op.exists():
                        opinion_paths.append(str(op))
                if len(plain_paths) == len(seeds_for_plot) and len(opinion_paths) == len(seeds_for_plot):
                    ds_out = str(Path(fig_dir) / f"ds_{plot_name_with_dataset}.png")
                    ds_cmd = [
                        sys.executable, "compute_decision_score.py",
                        "--plain", *plain_paths,
                        "--opinion", *opinion_paths,
                        "--out_plot", ds_out,
                    ]
                    print(f"[Plot] Running compute_decision_score.py -> {ds_out} ...")
                    subprocess.run(ds_cmd, cwd=str(mech_dir), timeout=300)
                else:
                    print(f"[Plot] Skip compute_decision_score ({plot_name_with_dataset}): missing pkl for some seeds", file=sys.stderr)
            elif not eval_mechanistic:
                print("[Plot] Skip compute_decision_score: eval_mechanistic=false", file=sys.stderr)
            else:
                print("[Plot] Skip compute_decision_score: logit_cot jobs not found", file=sys.stderr)

            # 3) plot_kl_divergence.py：多 seed 取平均，一张图（机理分支）
            if eval_mechanistic:
                kl_out = str(Path(fig_dir) / f"kl_divergence_{plot_name_with_dataset}.png")
                kl_cmd = [
                    sys.executable, "plot_kl_divergence.py",
                    "--plain_dir", f"output_inference/{_syco_dataset}/plain",
                    "--opinion_dir", f"output_inference/{_syco_dataset}/opinion_only",
                    "--out_plot", kl_out,
                    "--model_key", plot_output_name,
                    "--data_seeds", *[str(s) for s in seeds_for_plot],
                ]
                print(f"[Plot] Running plot_kl_divergence.py -> {kl_out} ...")
                subprocess.run(kl_cmd, cwd=str(mech_dir), timeout=300)
            else:
                print("[Plot] Skip plot_kl_divergence: eval_mechanistic=false", file=sys.stderr)

            print(f"[Plot] Done {plot_name_with_dataset}")
        print("Plot phase done: all combos, 1 figure per combo (avg over seeds)", seeds_for_plot)


if __name__ == "__main__":
    main()
