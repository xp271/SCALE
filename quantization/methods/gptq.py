"""GPTQ method builder (no overrides; uses official template as-is)."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "GPTQ",
    BaseMethod(
        MethodSpec(
            name="GPTQ",
            template_yml="configs/quantization/methods/GPTQ/gptq_w_only.yml",
            default_method_id_fmt="gptq_w{bit}",
        )
    ),
)
