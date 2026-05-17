"""Normalize the ``syco.data_seed`` config (None / int / list[int])."""
from __future__ import annotations


def normalize_data_seeds(raw) -> list[int | None]:
    """Return a list of seeds: single int -> [int]; None -> [None]; list -> int-cast list."""
    if raw is None:
        return [None]
    if isinstance(raw, list):
        return [s if s is None else int(s) for s in raw]
    return [int(raw)]
