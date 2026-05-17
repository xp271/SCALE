"""SmoothQuant method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "SmoothQuant",
    BaseMethod(
        MethodSpec(
            name="SmoothQuant",
            template_yml="configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml",
            default_method_id_fmt="smoothquant_w{bit}",
        )
    ),
)
