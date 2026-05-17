"""Registry of quantization method builders.

Each ``quantization/methods/<name>.py`` calls :func:`register` at import time
to expose its :class:`BaseMethod` instance. Use :func:`get_method` to look up
by config ``method`` name (e.g. ``"Awq"``).

The registry intentionally avoids importing :mod:`quantization.methods.base`
at top level: method modules import :func:`register` from here, and triggering
methods package load eagerly inside this module would introduce a circular
import. Callers should import :mod:`quantization` (whose ``__init__`` ensures
methods are loaded) or call :func:`_ensure_methods_loaded` explicitly.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantization.methods.base import BaseMethod

# 每个量化方法固定跑 4/6/8 bit（仅配置 method 时生效）
DEFAULT_WEIGHT_BITS = [4, 6, 8]

_REGISTRY: dict[str, "BaseMethod"] = {}
_METHODS_LOADED = False


def register(name: str, method: "BaseMethod") -> None:
    """Register a method builder under the canonical config name."""
    _REGISTRY[name] = method


def _ensure_methods_loaded() -> None:
    """Trigger import of the methods subpackage so registrations populate."""
    global _METHODS_LOADED
    if _METHODS_LOADED:
        return
    import quantization.methods  # noqa: F401  (side-effect: registrations)
    _METHODS_LOADED = True


def get_method(name: str) -> "BaseMethod | None":
    """Return the registered :class:`BaseMethod`, or ``None`` if unknown."""
    _ensure_methods_loaded()
    return _REGISTRY.get(name)


def list_registered_methods() -> list[str]:
    """Return registered method names (importing methods package triggers registration)."""
    _ensure_methods_loaded()
    return sorted(_REGISTRY.keys())


def parse_weight_bits_from_method_id(method_id: str) -> int | None:
    """从 method_id 解析权重量化比特数，如 rtn_w2 -> 2, gptq_w4 -> 4。若无 _wN 则返回 None。"""
    m = re.search(r"_w(\d+)$", method_id.strip())
    return int(m.group(1)) if m else None


def expand_methods_to_bits(methods: list) -> list[tuple[str, str, int]]:
    """将 methods 配置展开为 (method_name, method_id, weight_bits) 列表。

    仅配置 ``method`` 时固定跑 4/6/8 bit；显式给出 ``method_id`` 时按其后缀解析 bit。
    未注册的 method 会被跳过。
    """
    _ensure_methods_loaded()
    result: list[tuple[str, str, int]] = []
    for m in methods:
        method_name = m.get("method")
        method_id = m.get("method_id")
        method = _REGISTRY.get(method_name)
        if method is None:
            continue
        if method_id is not None:
            bits = parse_weight_bits_from_method_id(method_id)
            result.append((method_name, method_id, bits if bits is not None else 8))
        else:
            for b in DEFAULT_WEIGHT_BITS:
                result.append((method_name, method.default_method_id(b), b))
    return result
