"""DGQ method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "DGQ",
    BaseMethod(
        MethodSpec(
            name="DGQ",
            template_yml="configs/quantization/methods/DGQ/dgq_w_a.yml",
            default_method_id_fmt="dgq_w{bit}",
        )
    ),
)
