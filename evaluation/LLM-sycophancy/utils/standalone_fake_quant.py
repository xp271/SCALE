"""
Standalone inference-only FakeQuant linear layer for loading LightCompress fake_quant
checkpoints without depending on LightCompress. Reproduces EffcientFakeQuantLinear
forward (weight is pre-dequant; optional static activation quant from buf_*).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


def _act_quant_dequant_static(
    act: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    qmin: Optional[torch.Tensor],
    qmax: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-group static activation fake quant: quant then dequant. Matches LightCompress IntegerQuantizer."""
    org_shape = act.shape
    org_dtype = act.dtype
    # scales shape (num_groups,); act (..., in_features); group_size = in_features // num_groups
    num_groups = scales.shape[0]
    in_features = act.shape[-1]
    group_size = in_features // num_groups
    if group_size * num_groups != in_features:
        return act
    t = act.reshape(-1, group_size)
    scales = scales.to(act.device).to(act.dtype)
    zeros = zeros.to(act.device).to(act.dtype)
    # quant: round(x / scales + zeros), clamp if qmin/qmax present
    scaled = t / scales.unsqueeze(1) + zeros.unsqueeze(1)
    q = torch.round(scaled)
    if qmin is not None and qmax is not None:
        q = torch.clamp(q, qmin.item(), qmax.item())
    # dequant: (q - zeros) * scales
    out = (q - zeros.unsqueeze(1)) * scales.unsqueeze(1)
    out = out.reshape(org_shape).to(org_dtype)
    return out


class StandaloneFakeQuantLinear(nn.Module):
    """
    Inference-only layer: same forward as EffcientFakeQuantLinear when weight is
    pre-computed (w_qdq) and a_qdq is static (uses buf_act_scales_0, buf_act_zeros_0, etc.).
    No w_qdq/a_qdq callables; all from buffers.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", torch.empty(0))
        self.in_features = in_features
        self.out_features = out_features
        self._buffers_extra = buffers or {}
        for name, buf in self._buffers_extra.items():
            if name.startswith("buf_"):
                self.register_buffer(name, buf)
        self._has_act_quant = (
            hasattr(self, "buf_act_scales_0")
            and hasattr(self, "buf_act_zeros_0")
        )
        self.fp8_forward = getattr(weight, "dtype", torch.float32) == torch.float8_e4m3fn
        if self.fp8_forward and "weight_scale_inv" in self._buffers_extra:
            self.register_buffer("weight_scale_inv", self._buffers_extra["weight_scale_inv"])
            blk = self._buffers_extra.get("block_size")
            self.block_size = int(blk.item()) if blk is not None and hasattr(blk, "item") else 128
        else:
            self.block_size = 128

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_act_quant:
            scales = getattr(self, "buf_act_scales_0")
            zeros = getattr(self, "buf_act_zeros_0")
            qmax = getattr(self, "buf_act_qmax_0", None)
            qmin = getattr(self, "buf_act_qmin_0", None)
            x = _act_quant_dequant_static(x, scales, zeros, qmin, qmax)
        x = x.to(self.weight.dtype)
        if self.fp8_forward and hasattr(self, "weight_scale_inv"):
            try:
                from llmc.compression.quantization.module_utils import block_wise_fp8_forward_func
                bias = self.bias if self.bias.numel() > 0 else None
                y = block_wise_fp8_forward_func(
                    x, self.weight, self.weight_scale_inv, self.block_size, bias
                )
            except Exception:
                bias = self.bias if self.bias.numel() > 0 else None
                y = F.linear(x, self.weight.float(), bias)
        else:
            bias = self.bias if self.bias.numel() > 0 else None
            y = F.linear(x, self.weight, bias)
        return y


def _has_fake_quant_buffers(state_dict: Dict[str, torch.Tensor], prefix: str) -> bool:
    """True if this layer has any buf_* in state_dict (fake_quant layer)."""
    dot_prefix = prefix + "."
    for key in state_dict:
        if key == prefix or key.startswith(dot_prefix):
            rest = key[len(dot_prefix):] if key.startswith(dot_prefix) else key[len(prefix):]
            if rest.startswith("buf_"):
                return True
    return False


def _gather_buffers_for_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """Collect all buf_* and weight/bias for a given module prefix."""
    out = {}
    dot_prefix = prefix + "." if prefix else ""
    for key, value in state_dict.items():
        if not key.startswith(dot_prefix) and key != prefix:
            continue
        short = key[len(dot_prefix):] if key.startswith(dot_prefix) else key[len(prefix):].lstrip(".")
        if not short or "." in short:
            continue
        if short.startswith("buf_") or short in ("weight", "bias"):
            out[short] = value
    return out


def replace_linear_with_standalone_fake_quant(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
) -> int:
    """
    In-place replace nn.Linear submodules that have corresponding buf_* in state_dict
    with StandaloneFakeQuantLinear, and load their state from state_dict.
    Returns the number of Linear modules replaced.
    """
    replaced = 0
    for name, module in list(model.named_children()):
        full_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            if not _has_fake_quant_buffers(state_dict, full_prefix):
                continue
            bufs = _gather_buffers_for_prefix(state_dict, full_prefix)
            weight = bufs.get("weight")
            bias = bufs.get("bias")
            if weight is None:
                continue
            in_features = module.in_features
            out_features = module.out_features
            buffer_dict = {k: v for k, v in bufs.items() if k not in ("weight", "bias")}
            new_module = StandaloneFakeQuantLinear(
                weight=weight,
                bias=bias,
                in_features=in_features,
                out_features=out_features,
                buffers=buffer_dict,
            )
            setattr(model, name, new_module)
            replaced += 1
        else:
            replaced += replace_linear_with_standalone_fake_quant(module, state_dict, full_prefix)
    return replaced


def load_fake_quant_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load full state_dict from a HuggingFace-style model directory (safetensors or bin)."""
    from pathlib import Path
    import os
    path = Path(model_dir)
    state_dict = {}
    # Prefer safetensors
    st_files = list(path.glob("*.safetensors"))
    if st_files:
        try:
            from safetensors.torch import load_file
            for f in st_files:
                state_dict.update(load_file(str(f)))
            return state_dict
        except Exception:
            pass
    bin_file = path / "pytorch_model.bin"
    if bin_file.exists():
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
        return state_dict
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {model_dir}")


def _count_buf_prefixes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Count distinct module prefixes that have at least one buf_* tensor key."""
    prefixes = set()
    for k in state_dict:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p.startswith("buf_"):
                prefixes.add(".".join(parts[:i]))
                break
    return len(prefixes)


def load_model_for_fake_quant(
    fake_quant_dir: str,
    base_model_name: str,
    device: Optional[Any] = None,
    device_map: Optional[str] = None,
    token: Optional[str] = None,
    debug_inference: bool = False,
):
    """
    Load a model that was saved as fake_quant by LightCompress: build base model,
    replace Linear layers that have buf_* with StandaloneFakeQuantLinear, then
    load state_dict from fake_quant_dir.
    If config.tie_word_embeddings is True, always calls tie_weights() after load so
    lm_head stays tied to input embeddings when the checkpoint omits lm_head.weight.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kw = {"trust_remote_code": True}
    if token:
        tokenizer_kw["token"] = token
    # Prefer local tokenizer in fake_quant_dir to avoid network dependency.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            fake_quant_dir,
            fix_mistral_regex=True,
            local_files_only=True,
            **tokenizer_kw,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            fake_quant_dir,
            local_files_only=True,
            **tokenizer_kw,
        )
    except Exception:
        # Fallback to base model path/id if fake_quant_dir misses tokenizer files.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                fix_mistral_regex=True,
                **tokenizer_kw,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tokenizer_kw)

    model_kw = {"trust_remote_code": True}
    if token:
        model_kw["token"] = token
    if device_map:
        model_kw["device_map"] = device_map
    # IMPORTANT: initialize from base pretrained weights first, then overlay fake_quant state.
    # This avoids random initialization for any keys not present in fake_quant state dict.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            local_files_only=True,
            **model_kw,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kw)

    state_dict = load_fake_quant_state_dict(fake_quant_dir)
    buf_keys = sum(1 for k in state_dict if "buf_" in k)
    n_replaced = replace_linear_with_standalone_fake_quant(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    # HuggingFace CausalLM often stores one matrix when tie_word_embeddings=True; LightCompress export
    # fake_quant often has only model.embed_tokens.weight, not lm_head.weight. load reports missing,
    # without re-tie, lm_head still points at pre-load tensor, detached from updated embedding.
    # Default matches HF tie_embeddings_and_encoder_decoder: True when config omits it
    tie_emb = getattr(model.config, "tie_word_embeddings", True)
    if hasattr(model.config, "get_text_config"):
        try:
            tie_emb = getattr(
                model.config.get_text_config(decoder=True),
                "tie_word_embeddings",
                tie_emb,
            )
        except Exception:
            pass
    if tie_emb and hasattr(model, "tie_weights"):
        model.tie_weights()

    missing_warn = [
        k
        for k in missing
        if not (tie_emb and k in ("lm_head.weight", "lm_head.bias"))
    ]
    if missing_warn or unexpected:
        print(
            "[fake_quant][warn] load_state_dict(strict=False): "
            f"missing_unresolved={len(missing_warn)}, unexpected={len(unexpected)}"
        )
        if missing_warn:
            print(f"[fake_quant][warn] sample missing keys: {missing_warn[:10]}")
        if unexpected:
            print(f"[fake_quant][warn] sample unexpected keys: {unexpected[:10]}")
    elif missing and tie_emb:
        print(
            "[fake_quant] checkpoint omitted tied lm_head key(s); "
            "called tie_weights() so lm_head matches input embeddings."
        )
    if debug_inference:
        pref = "[syco_debug][fake_quant_load]"
        tied_omitted = [
            k for k in missing if tie_emb and k in ("lm_head.weight", "lm_head.bias")
        ]
        print(
            f"{pref} state_dict_keys={len(state_dict)} buf_*_key_count={buf_keys} "
            f"distinct_linear-ish_buf_prefixes~={_count_buf_prefixes(state_dict)} "
            f"replaced_linear_with_standalone={n_replaced}"
        )
        print(
            f"{pref} load_state_dict missing_raw={len(missing)} "
            f"missing_unresolved={len(missing_warn)} "
            f"tied_head_omitted_in_ckpt={tied_omitted} unexpected={len(unexpected)}"
        )
        try:
            p0 = next(model.parameters())
            print(f"{pref} first_param device={p0.device} dtype={p0.dtype}")
        except StopIteration:
            print(f"{pref} first_param: (no parameters)")
        for probe in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ):
            w = state_dict.get(probe)
            if w is None:
                continue
            wf = w.float()
            print(
                f"{pref} {probe} shape={tuple(w.shape)} dtype={w.dtype} "
                f"norm={float(wf.norm())} any_nan={bool(torch.isnan(wf).any())}"
            )
    if device is not None and not device_map:
        model = model.to(device)
    elif device_map:
        # Replaced layers got tensors from state_dict on CPU; move whole model to device_map so all tensors match
        target = torch.device(device_map)
        model = model.to(target)
    model.eval()
    return model, tokenizer
