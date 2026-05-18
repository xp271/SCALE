"""AWQ method builder.
"""
from __future__ import annotations

from quantization.methods.base import BaseMethod, MethodSpec
from quantization.registry import register

AWQ_IGNORE_FIRST_N_BLOCKS = 5
AWQ_IGNORE_LAYER_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


class AwqMethod(BaseMethod):
    def post_process(self, run: dict, *, weight_bits: int | None) -> None:
        run["ignored_layers"] = {
            "block_ids": list(range(AWQ_IGNORE_FIRST_N_BLOCKS)),
            "layer_names": list(AWQ_IGNORE_LAYER_NAMES),
        }


register(
    "Awq",
    AwqMethod(
        MethodSpec(
            name="Awq",
            template_yml="configs/quantization/methods/Awq/awq_w_only.yml",
            default_method_id_fmt="awq_w{bit}",
        )
    ),
)
