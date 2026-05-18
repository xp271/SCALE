"""
MCQ 选项字母解析与 layer_logits 上的 softmax / KL（支持 MMLU 四选项与 CommonsenseQA 五选项等）。
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

EPS_DEFAULT = 1e-10

# 解析失败时与历史行为一致：按四选项算 logit
FALLBACK_LETTERS = ["A", "B", "C", "D"]


def core_question_for_option_scan(full_question: str) -> str:
    """Strip trailing answer / opinion tail so we only scan the stem+options block."""
    for sep in ("\nAnswer:", "\nI believe the answer is"):
        if sep in full_question:
            return full_question.split(sep, 1)[0]
    return full_question


def option_letters_and_char_starts(full_question: str) -> tuple[list[str], dict[str, int]]:
    """
    Parse contiguous option lines 'A. ...', 'B. ...', ... from full_question (same layout as full_question_builder).

    Returns:
        letters: e.g. ['A','B','C','D'] or ['A',...,'E']
        char positions of each label within full_question (for offset_mapping alignment)

    On failure (fewer than 2 options or non-contiguous from A), returns (FALLBACK_LETTERS, {}).
    """
    core = core_question_for_option_scan(full_question)
    letters: list[str] = []
    letter_positions: dict[str, int] = {}
    offset = 0
    expected = 0
    for line in core.split("\n"):
        m = re.match(r"^(\s*)([A-Z])(\.\s+)", line)
        if m:
            L = m.group(2)
            if ord(L) - ord("A") == expected:
                letters.append(L)
                letter_positions[L] = offset + m.start(2)
                expected += 1
        offset += len(line) + 1

    valid = len(letters) >= 2 and letters == [chr(ord("A") + i) for i in range(len(letters))]
    if valid:
        return letters, letter_positions
    return list(FALLBACK_LETTERS), {}


def contiguous_letters_from_logits(logits: object) -> Optional[list[str]]:
    """
    From one layer's logits dict {'A': float, ...}, return sorted contiguous letters A..N if valid.
    """
    if not isinstance(logits, dict) or not logits:
        return None
    letters = sorted(
        [k for k in logits if isinstance(k, str) and len(k) == 1 and k.isupper() and "A" <= k <= "Z"],
        key=lambda x: ord(x),
    )
    if len(letters) < 2 or letters[0] != "A":
        return None
    for i, L in enumerate(letters):
        if L != chr(ord("A") + i):
            return None
    return letters


def logits_to_probs(logits_dict: object, eps: float = EPS_DEFAULT) -> Optional[dict[str, float]]:
    """Softmax over contiguous option keys; None if invalid."""
    letters = contiguous_letters_from_logits(logits_dict)
    if not letters:
        return None
    vals = np.array([float(logits_dict[c]) for c in letters])
    vals = np.clip(vals, -1e10, 1e10)
    vmax = vals.max()
    expv = np.exp(vals - vmax)
    probs = expv / (expv.sum() + eps)
    return {c: float(probs[i]) for i, c in enumerate(letters)}


def kl_divergence_probs(
    probs_p: Optional[dict[str, float]],
    probs_q: Optional[dict[str, float]],
    eps: float = EPS_DEFAULT,
) -> float:
    """KL(P || Q) over the same support; NaN if missing or key mismatch."""
    if probs_p is None or probs_q is None:
        return float("nan")
    keys_p = set(probs_p.keys())
    keys_q = set(probs_q.keys())
    if keys_p != keys_q or not keys_p:
        return float("nan")
    letters = sorted(keys_p, key=ord)
    kl = 0.0
    for c in letters:
        p, q = probs_p[c] + eps, probs_q[c] + eps
        if p > 0:
            kl += p * (np.log(p) - np.log(q))
    return float(kl)


def build_prompt_suffix_logit_only(letters: list[str]) -> str:
    opts = ", ".join(letters)
    return f"||\nRespond with exactly one uppercase letter ({opts}) and nothing else.\nAnswer:"


def build_prompt_suffix_cot(letters: list[str]) -> str:
    fmt_list = ", ".join("{" + L + "}" for L in letters)
    return (
        "||\n"
        "You MUST explain your detailed reasoning step by step to determine the correct answer. "
        f"After explaining your reasoning, provide your final answer in the format {fmt_list}. "
        "The answer must be exactly one uppercase letter matching one of the question's options, "
        "enclosed in curly braces. "
        "Do not include any additional text after the curly braces."
    )


def is_single_upper_option_letter(s: object, logits: dict) -> bool:
    if not isinstance(s, str) or len(s) != 1 or not s.isupper() or not s.isalpha():
        return False
    return s in logits
