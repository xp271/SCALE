"""QuaRot method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "QuaRot",
    BaseMethod(
        MethodSpec(
            name="QuaRot",
            template_yml="configs/quantization/methods/QuaRot/quarot_w_a.yml",
            default_method_id_fmt="quarot_w{bit}",
        )
    ),
)
