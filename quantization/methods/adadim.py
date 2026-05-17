"""AdaDim method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "AdaDim",
    BaseMethod(
        MethodSpec(
            name="AdaDim",
            template_yml="configs/quantization/methods/AdaDim/adadim_w_a.yml",
            default_method_id_fmt="adadim_w{bit}",
        )
    ),
)
