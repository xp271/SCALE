"""TesseraQ method builder."""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

register(
    "TesseraQ",
    BaseMethod(
        MethodSpec(
            name="TesseraQ",
            template_yml="configs/quantization/methods/Tesseraq/tesseraq_w_only.yml",
            default_method_id_fmt="tesseraq_w{bit}",
        )
    ),
)
