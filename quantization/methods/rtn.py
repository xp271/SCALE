"""RTN method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "RTN",
    BaseMethod(
        MethodSpec(
            name="RTN",
            template_yml="configs/quantization/methods/RTN/rtn_w_only.yml",
            default_method_id_fmt="rtn_w{bit}",
        )
    ),
)
