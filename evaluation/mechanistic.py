"""Entry helpers for the mechanistic_analysis experiment.

Thin wrapper that re-exports :func:`build_mechanistic_jobs` and provides a
default subscript directory pointer. The actual subprocess invocation goes
through :func:`evaluation.runner.run_syco_eval`.
"""
from __future__ import annotations

from evaluation.job_builder import DIR_MECH_SCRIPT as MECHANISTIC_SCRIPT_DIR
from evaluation.job_builder import build_mechanistic_jobs

__all__ = ["build_mechanistic_jobs", "MECHANISTIC_SCRIPT_DIR"]
