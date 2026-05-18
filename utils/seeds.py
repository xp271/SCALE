"""Normalize the ``syco.data_seed`` config (None / int / list[int]) and RNG helpers."""
from __future__ import annotations

import random


def generate_eval_data_seeds(master_seed: int, n: int) -> list[int]:
    """用固定 ``master_seed`` 的 PRNG 生成 ``n`` 个互不相同的数据集随机种子（用于 build_lib / 评估 / 绘图平均）。

    取值范围为 ``[0, 2**31)``，生成顺序稳定可复现。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    rng = random.Random(master_seed)
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < n:
        x = rng.randrange(0, 2**31)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_data_seeds(raw) -> list[int | None]:
    """Return a list of seeds: single int -> [int]; None -> [None]; list -> int-cast list."""
    if raw is None:
        return [None]
    if isinstance(raw, list):
        return [s if s is None else int(s) for s in raw]
    return [int(raw)]
