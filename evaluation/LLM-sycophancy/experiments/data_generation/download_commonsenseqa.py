"""
Download tau/commonsense_qa from Hugging Face, convert to data_generation schema, save as pkl.
Required columns: question, subject, choices (list[str]), answer (0-based int).

Default split is validation (has answerKey, suitable for supervised pipeline). test often has no labels;
rows without answers are skipped with a warning.

Example:
  python download_commonsenseqa.py --output raw_data/commonsenseqa_raw.pkl
  python download_commonsenseqa.py --split train --output raw_data/commonsenseqa_train_raw.pkl
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def _normalize_choice_pairs(choices) -> list[tuple[str, str]] | None:
    """Return list of (label, text) sorted by label, or None if invalid."""
    if choices is None:
        return None
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels, texts = choices["label"], choices["text"]
        if not isinstance(labels, (list, tuple)) or not isinstance(texts, (list, tuple)):
            return None
        if len(labels) != len(texts):
            return None
        pairs = [(str(l).strip().upper(), str(t)) for l, t in zip(labels, texts)]
        pairs.sort(key=lambda x: x[0])
        return pairs
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        pairs = []
        for c in choices:
            if not isinstance(c, dict):
                return None
            lab = c.get("label")
            txt = c.get("text")
            if lab is None or txt is None:
                return None
            pairs.append((str(lab).strip().upper(), str(txt)))
        pairs.sort(key=lambda x: x[0])
        return pairs
    return None


def answer_key_to_index(answer_key, n_choices: int) -> int | None:
    if answer_key is None:
        return None
    if isinstance(answer_key, int):
        if 0 <= answer_key < n_choices:
            return answer_key
        return None
    s = str(answer_key).strip().upper()
    if len(s) != 1 or not s.isalpha():
        return None
    idx = ord(s) - ord("A")
    if 0 <= idx < n_choices:
        return idx
    return None


def subject_from_item(item, fallback: str = "commonsense") -> str:
    concept = item.get("question_concept")
    if concept is None:
        return fallback
    try:
        if isinstance(concept, (float, int)) and pd.isna(concept):
            return fallback
    except (TypeError, ValueError):
        pass
    s = str(concept).strip()
    return s if s else fallback


def main():
    parser = argparse.ArgumentParser(
        description="Download CommonsenseQA (tau/commonsense_qa) and save as raw pkl.",
        epilog=(
            "Recommended split: validation or train (has answerKey). test often has no labels; script drops unlabeled rows.\n"
            "Generate prefixes: python generate_prefixes.py --raw_file raw_data/commonsenseqa_raw.pkl "
            "--output_name_prefix academic_prefix_commonsenseqa\n"
            "MMLU download: python download_mmlu.py --output raw_data/mmlu_raw.pkl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw_data/commonsenseqa_raw.pkl",
        help="Output pkl path (relative to cwd)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Data split; default validation. test may lack answerKey; unlabeled samples are dropped.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tau/commonsense_qa",
        help="Hugging Face dataset id; default tau/commonsense_qa",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install first: pip install datasets")
        raise

    print(f"Loading {args.dataset} from Hugging Face (split={args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split)

    rows = []
    dropped_no_answer = 0
    dropped_bad_choices = 0

    for item in ds:
        question = item.get("question")
        if question is None:
            continue
        question = str(question)

        pairs = _normalize_choice_pairs(item.get("choices"))
        if not pairs:
            dropped_bad_choices += 1
            continue
        choice_texts = [p[1] for p in pairs]
        n = len(choice_texts)
        if n == 0:
            dropped_bad_choices += 1
            continue

        ak = item.get("answerKey")
        answer_index = answer_key_to_index(ak, n)
        if answer_index is None:
            dropped_no_answer += 1
            continue

        rows.append(
            {
                "question": question,
                "subject": subject_from_item(item),
                "choices": choice_texts,
                "answer": answer_index,
            }
        )

    if dropped_no_answer:
        print(
            f"Warning: skipped {dropped_no_answer} rows with invalid answerKey or option count mismatch "
            f"(common on split={args.split}, especially test)."
        )
    if dropped_bad_choices:
        print(f"Warning: skipped {dropped_bad_choices} rows with unparseable choices format.")

    df = pd.DataFrame(rows)
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_pickle(out_path)
    print(f"Saved {len(df)} rows to {os.path.abspath(out_path)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
