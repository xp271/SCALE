"""LlmInt8 method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "LlmInt8",
    BaseMethod(
        MethodSpec(
            name="LlmInt8",
            template_yml="configs/quantization/methods/LlmInt8/llmint8_w_only.yml",
            default_method_id_fmt="llmint8_w{bit}",
        )
    ),
)
