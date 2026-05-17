"""QUIK method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "QUIK",
    BaseMethod(
        MethodSpec(
            name="QUIK",
            template_yml="configs/quantization/methods/QUIK/quik_w_a.yml",
            default_method_id_fmt="quik_w{bit}",
        )
    ),
)
