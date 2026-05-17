"""OsPlus method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "OsPlus",
    BaseMethod(
        MethodSpec(
            name="OsPlus",
            template_yml="configs/quantization/methods/OsPlus/osplus_w_a.yml",
            default_method_id_fmt="osplus_w{bit}",
        )
    ),
)
