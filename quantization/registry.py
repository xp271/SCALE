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

# Each method runs 4/6/8 bit when only method is configured
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
    """Parse weight bits from method_id, e.g. rtn_w2 -> 2, gptq_w4 -> 4. None if no _wN suffix."""
    m = re.search(r"_w(\d+)$", method_id.strip())
    return int(m.group(1)) if m else None


def expand_methods_to_bits(methods: list) -> list[tuple[str, str, int]]:
    """Expand methods config to list of (method_name, method_id, weight_bits).

    When only ``method`` is set, runs 4/6/8 bit; explicit ``method_id`` uses suffix for bit.
    Unregistered methods are skipped.

    Note: run_pipeline main entry uses CLI single bit width and
    :func:`resolve_method_quant_combo`; this remains for other batch scripts.
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


def resolve_method_quant_combo(method_cfg: dict, weight_bits: int) -> tuple[str, str, int]:
    """Resolve unique (method_name, method_id, weight_bits) from yaml entry and CLI ``--bits``.

    - If ``method_id`` has ``_wN`` suffix, ``N`` must match ``weight_bits``.
    - If ``method_id`` lacks ``_wN``, error (drop field and use ``--bits`` only, or use ``*_wN``).
    - If no ``method_id``, use ``default_method_id(weight_bits)``.
    """
    _ensure_methods_loaded()
    method_name = method_cfg.get("method")
    if not method_name:
        raise ValueError("methods config missing method field")
    method = _REGISTRY.get(method_name)
    if method is None:
        raise ValueError(f"unknown quantization method: {method_name!r}")
    fixed_id = method_cfg.get("method_id")
    if fixed_id is not None:
        parsed = parse_weight_bits_from_method_id(fixed_id)
        if parsed is not None:
            if parsed != weight_bits:
                raise ValueError(
                    f"yaml method_id={fixed_id!r} is W{parsed}, inconsistent with --bits {weight_bits}; "
                    "fix bit width or remove method_id"
                )
            return (method_name, fixed_id, weight_bits)
        raise ValueError(
            f"yaml method_id={fixed_id!r} has no parseable bit width (need *_wN suffix); "
            f"remove method_id and use --bits {weight_bits} only, or use e.g. {method.default_method_id(weight_bits)!r}"
        )
    return (method_name, method.default_method_id(weight_bits), weight_bits)
