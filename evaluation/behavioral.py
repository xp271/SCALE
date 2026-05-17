"""Entry helpers for the behavioral_analysis experiment.

Thin wrapper that re-exports :func:`build_behavioral_jobs` and provides a
default subscript directory pointer. The actual subprocess invocation goes
through :func:`evaluation.runner.run_syco_eval`.
"""
from __future__ import annotations

from evaluation.job_builder import build_behavioral_jobs
from utils.paths import DIR_SYCO_SCRIPT as BEHAVIORAL_SCRIPT_DIR

__all__ = ["build_behavioral_jobs", "BEHAVIORAL_SCRIPT_DIR"]
