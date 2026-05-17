"""HQQ method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "HQQ",
    BaseMethod(
        MethodSpec(
            name="HQQ",
            template_yml="configs/quantization/methods/HQQ/hqq_w_only.yml",
            default_method_id_fmt="hqq_w{bit}",
        )
    ),
)
