"""Path helpers for syco eval inputs / expected pkl outputs."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_input_path(filename: str, repo_root: Path) -> str:
    """Resolve a syco lib pkl path relative to the LLM-sycophancy repo root."""
    if not filename or os.path.isabs(filename):
        return filename
    return str((repo_root / filename).resolve())


def resolve_seeded_input_path(filename: str, repo_root: Path, data_seed: int | None) -> Path:
    """Insert data_seed suffix into input pkl filename per run_syco rules."""
    p = Path(resolve_input_path(filename, repo_root))
    if data_seed is None:
        return p
    return p.with_name(f"{p.stem}_{data_seed}{p.suffix}")


def expected_syco_pkl_path(
    script_dir: Path,
    eval_job: dict,
    model_path: Path,
    dataset: str,
    *,
    data_seed: int | None = None,
    model_output_name: str | None = None,
    output_base_override: str | None = None,
) -> Path:
    """Match run_syco.py / run_syco_logit_cot.py save paths; infer output when not capture_output.

    ``output_base_override`` is ``--output_base``; if None, script defaults apply
    (behavioral: ``output``; mechanistic: ``output_inference``).
    """
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
        out_base = output_base_override or "output_inference"
        inference_mode = kwargs.get("inference_mode", "logit_only")
        inference_layer = kwargs.get("inference_layer", "all")
        mode_str = "cot" if inference_mode == "logit_and_cot" else "logit"
        parts = [out_base, dataset, question_type]
        if prefix_type:
            parts.extend([prefix_type, prefix_subtype, academic_level])
        rel_dir = os.path.join(*[p for p in parts if p])
        fname = f"{basename}_{mode_str}_{inference_layer}{seed_suffix}.pkl"
    else:
        out_base = output_base_override or "output"
        parts = [out_base, dataset, question_type]
        if prefix_type:
            parts.extend([prefix_type, prefix_subtype, academic_level])
        rel_dir = os.path.join(*[p for p in parts if p])
        fname = f"{basename}{seed_suffix}.pkl"
    rel_path = Path(rel_dir)
    if rel_path.is_absolute():
        return (rel_path / fname).resolve()
    return (script_dir / rel_dir / fname).resolve()
