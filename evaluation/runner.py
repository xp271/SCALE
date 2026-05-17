"""Run syco eval subprocesses (run_syco.py / run_syco_logit_cot.py).

This module exposes:

- :func:`run_syco_eval`: single-job subprocess wrapper.
- :func:`eval_model_against_jobs`: high-level loop that walks the eval-jobs x
  data-seeds matrix for one model, skipping evaluations whose output pkl is
  already on disk. Consolidates the FP and Quant inner loops that used to be
  duplicated in the monolithic pipeline.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from evaluation.accuracy import run_compute_accuracy
from evaluation.paths import (
    expected_syco_pkl_path,
    resolve_input_path,
    resolve_seeded_input_path,
)
from utils.env import truthy_inference_debug

TIMEOUT_SYCO_RUN = 3600 * 24


def _inject_runtime_kwargs(ej: dict, syco_args: dict) -> dict:
    """Add answer_max_new_tokens / debug_inference / save_option_hs / stream_hs_dir per script."""
    ej = dict(ej)
    if ej.get("script") == "run_syco.py":
        kwd = dict(ej.get("kwargs", {}))
        amnt = syco_args.get("answer_max_new_tokens")
        if amnt is not None:
            kwd["answer_max_new_tokens"] = int(amnt)
        if truthy_inference_debug(syco_args.get("debug_inference")):
            kwd["debug_inference"] = True
        ej["kwargs"] = kwd
    elif ej.get("script") == "run_syco_logit_cot.py":
        kwd = dict(ej.get("kwargs", {}))
        if syco_args.get("save_option_hs"):
            kwd["save_option_hs"] = True
        shd = syco_args.get("stream_hs_dir")
        if shd:
            kwd["stream_hs_dir"] = str(shd)
        ej["kwargs"] = kwd
    return ej


def run_syco_eval(
    model_path: Path,
    syco_repo_root: Path,
    eval_job: dict,
    *,
    device: str = "auto",
    dataset: str = "mmlu",
    full_question_column: str = "full_question",
    max_retries: int = 3,
    base_model_name: str | None = None,
    data_seed: int | None = None,
    model_output_name: str | None = None,
    cuda_visible_devices: str | None = None,
    output_base: str | None = None,
) -> str | None:
    """Run one syco eval (run_syco.py or run_syco_logit_cot.py). Return output pkl path if produced."""
    script_name = eval_job["script"]
    script_dir = syco_repo_root / eval_job["dir"]
    kwargs = dict(eval_job["kwargs"])
    input_path = resolve_input_path(kwargs.pop("input_filename", ""), syco_repo_root)
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
    if output_base:
        cmd.extend(["--output_base", output_base])
    for k, v in kwargs.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            continue
        cmd.extend([f"--{k}", str(v)])
    expected_pkl = expected_syco_pkl_path(
        script_dir,
        eval_job,
        model_path,
        dataset,
        data_seed=data_seed,
        model_output_name=model_output_name,
        output_base_override=output_base,
    )
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
        print(
            f"{script_name} timed out after {TIMEOUT_SYCO_RUN}s (eval {eval_job.get('tag', '')}), skipping.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"{script_name} failed: {e}", file=sys.stderr)
        return None


def _output_base_for_job(eval_job: dict, output_base_map: dict[str, str] | None) -> str | None:
    """Resolve which absolute output_base to pass to the subprocess for this job."""
    if not output_base_map:
        return None
    if eval_job["script"] == "run_syco_logit_cot.py":
        return output_base_map.get("mechanistic")
    return output_base_map.get("behavioral")


def eval_model_against_jobs(
    *,
    model_path: Path,
    syco_repo: Path,
    eval_jobs: Iterable[dict],
    data_seeds: list[int | None],
    syco_args: dict,
    syco_device: str,
    dataset: str,
    model_output_name: str,
    log_tag: str,                       # 例如 "Syco FP" / "Syco" / "Syco QAT"
    cuda_visible_devices: str | None,
    base_model_name: str | None,
    model_id: str,
    method_id: str,
    aggregate_csv: bool,
    output_base_map: dict[str, str] | None = None,
    accuracy_script_dir: Path | None = None,
) -> list[dict]:
    """Run all (data_seed × eval_job) combinations for one model_path.

    Returns a list of accuracy rows (one per data_seed with a for_aggregate pkl).
    Existing output pkl files short-circuit re-evaluation; missing seeded inputs
    cause the specific job to be skipped with a warning.

    ``output_base_map`` may contain ``{"behavioral": "<abs path>", "mechanistic":
    "<abs path>"}`` to redirect outputs via the new ``--output_base`` flag in
    run_syco{,_logit_cot}.py (introduced by the folder redesign).
    """
    eval_jobs = list(eval_jobs)
    full_question_column = syco_args.get("full_question_column", "full_question")
    max_retries = syco_args.get("max_retries", 3)
    rows: list[dict] = []
    for data_seed in data_seeds:
        pkl_for_aggregate: str | None = None
        for eval_job in eval_jobs:
            ej = _inject_runtime_kwargs(eval_job, syco_args)
            output_base = _output_base_for_job(ej, output_base_map)
            expected_eval_pkl = expected_syco_pkl_path(
                syco_repo / ej["dir"],
                ej,
                model_path,
                dataset,
                data_seed=data_seed,
                model_output_name=model_output_name,
                output_base_override=output_base,
            )
            if expected_eval_pkl.exists():
                print(
                    f"[{log_tag}] {model_id} x {method_id} (seed={data_seed}) -> "
                    f"skip {ej['tag']} (already exists: {expected_eval_pkl.name})"
                )
                if eval_job.get("for_aggregate"):
                    pkl_for_aggregate = str(expected_eval_pkl)
                continue
            input_filename = str(ej.get("kwargs", {}).get("input_filename", "")).strip()
            if input_filename:
                resolved_input = resolve_seeded_input_path(input_filename, syco_repo, data_seed)
                if not resolved_input.exists():
                    print(
                        f"[{log_tag}] {model_id} x {method_id} (seed={data_seed}) -> "
                        f"skip {ej['tag']} (missing input: {resolved_input})",
                        file=sys.stderr,
                    )
                    continue
            print(f"[{log_tag}] {model_id} x {method_id} (seed={data_seed}) -> {ej['tag']}")
            pkl_path = run_syco_eval(
                model_path,
                syco_repo,
                ej,
                device=syco_device,
                dataset=dataset,
                full_question_column=full_question_column,
                max_retries=max_retries,
                base_model_name=base_model_name,
                data_seed=data_seed,
                model_output_name=model_output_name,
                cuda_visible_devices=cuda_visible_devices,
                output_base=output_base,
            )
            if eval_job.get("for_aggregate") and pkl_path:
                pkl_for_aggregate = pkl_path
        if aggregate_csv and pkl_for_aggregate and accuracy_script_dir is not None:
            metrics = run_compute_accuracy(pkl_for_aggregate, accuracy_script_dir)
            if metrics:
                rows.append(
                    {
                        "model_id": model_id,
                        "method_id": method_id,
                        "data_seed": data_seed,
                        "pkl": pkl_for_aggregate,
                        **metrics,
                    }
                )
    return rows
