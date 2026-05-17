"""NormTweaking method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "NormTweaking",
    BaseMethod(
        MethodSpec(
            name="NormTweaking",
            template_yml="configs/quantization/methods/NormTweaking/ntweak_w_only.yml",
            default_method_id_fmt="normtweaking_w{bit}",
        )
    ),
)
