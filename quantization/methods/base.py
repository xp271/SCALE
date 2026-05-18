"""Shared base class for per-method LightCompress yml builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quantization.yml_builder import fill_template


@dataclass(frozen=True)
class MethodSpec:
    name: str               # e.g. "Awq" / "GPTQ"; must match pipeline_config.yaml methods.method
    template_yml: str       # template path relative to LightCompress root
    default_method_id_fmt: str  # e.g. "awq_w{bit}"


class BaseMethod:
    """Per-method yml builder. Subclasses override ``post_process`` for overrides."""

    spec: MethodSpec

    def __init__(self, spec: MethodSpec) -> None:
        self.spec = spec

    def template_path(self, llmc_root: Path) -> Path:
        return llmc_root / self.spec.template_yml

    def default_method_id(self, weight_bits: int) -> str:
        return self.spec.default_method_id_fmt.format(bit=weight_bits)

    def build_run_yml(
        self,
        llmc_root: Path,
        *,
        model_path: str,
        model_type: str,
        save_path: Path,
        calib_root: Path,
        eval_root: Path,
        base_seed: int,
        weight_bits: int | None = None,
        calib_auto_download: bool = False,
        eval_auto_download: bool = False,
    ) -> dict:
        run = fill_template(
            self.template_path(llmc_root),
            model_path=model_path,
            model_type=model_type,
            save_path=save_path,
            calib_root=calib_root,
            eval_root=eval_root,
            base_seed=base_seed,
            calib_auto_download=calib_auto_download,
            eval_auto_download=eval_auto_download,
            weight_bits=weight_bits,
        )
        self.post_process(run, weight_bits=weight_bits)
        return run

    def post_process(self, run: dict, *, weight_bits: int | None) -> None:
        """Hook for method-specific overrides (default: no-op)."""
        return None
