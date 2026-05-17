"""OmniQuant method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "OmniQuant",
    BaseMethod(
        MethodSpec(
            name="OmniQuant",
            template_yml="configs/quantization/methods/OmniQuant/omniq_w_only.yml",
            default_method_id_fmt="omniquant_w{bit}",
        )
    ),
)
