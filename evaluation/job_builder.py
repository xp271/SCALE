"""Build the list of syco eval jobs to run against a model.

A job is a small dict ``{script, dir, tag, for_aggregate, kwargs}`` consumed by
:mod:`evaluation.runner`. The two upstream experiments live under different
LLM-sycophancy subdirs:

- behavioral (``experiments/behavioral_analysis``): ``run_syco.py`` with plain
  / opinion_only / authority levels / behavior prefix.
- mechanistic (``experiments/mechanistic_analysis``): ``run_syco_logit_cot.py``
  for logit-only inference on plain / opinion_only.
"""
from __future__ import annotations

from utils.paths import DIR_SYCO_SCRIPT

DIR_MECH_SCRIPT = "experiments/mechanistic_analysis"


def build_behavioral_jobs(
    data_slug: str,
    *,
    eval_authority_advanced: bool = False,
    eval_behavior_prefix: bool = False,
    behavior_input_filename: str | None = None,
) -> list[dict]:
    """Plain / Opinion-Only (+ optional Authority three-level, Behavior-prefix)."""
    plain_in = f"lib/plain/{data_slug}_plain.pkl"
    opinion_in = f"lib/opinion_only/prefix/{data_slug}_opinion_only.pkl"
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
    if eval_authority_advanced:
        # Run all three levels for fig2 (Beginner / Intermediate / Advanced)
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


def build_mechanistic_jobs(data_slug: str) -> list[dict]:
    """logit_only inference on plain / opinion_only (used for KL/DS plots)."""
    plain_in = f"lib/plain/{data_slug}_plain.pkl"
    opinion_in = f"lib/opinion_only/prefix/{data_slug}_opinion_only.pkl"
    return [
        {
            "script": "run_syco_logit_cot.py",
            "dir": DIR_MECH_SCRIPT,
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
            "dir": DIR_MECH_SCRIPT,
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


def build_syco_eval_jobs(
    data_slug: str,
    *,
    eval_mechanistic: bool = True,
    eval_authority_advanced: bool = False,
    eval_behavior_prefix: bool = False,
    behavior_input_filename: str | None = None,
) -> list[dict]:
    """Compose behavioral + (optional) mechanistic jobs into a single list."""
    jobs = build_behavioral_jobs(
        data_slug,
        eval_authority_advanced=eval_authority_advanced,
        eval_behavior_prefix=eval_behavior_prefix,
        behavior_input_filename=behavior_input_filename,
    )
    # Order: plain / opinion_only / logit_cot_* / authority_* / behavior_prefix
    # Legacy: mechanistic jobs right after plain/opinion, then authority/behavior.
    if eval_mechanistic:
        mech = build_mechanistic_jobs(data_slug)
        # Insert after plain+opinion, before authority/behavior
        head = [j for j in jobs if j["tag"] in ("plain", "opinion_only")]
        tail = [j for j in jobs if j["tag"] not in ("plain", "opinion_only")]
        jobs = head + mech + tail
    return jobs
