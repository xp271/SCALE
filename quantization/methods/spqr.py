"""SpQR method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "SpQR",
    BaseMethod(
        MethodSpec(
            name="SpQR",
            template_yml="configs/quantization/methods/SpQR/spqr_w_only.yml",
            default_method_id_fmt="spqr_w{bit}",
        )
    ),
)
