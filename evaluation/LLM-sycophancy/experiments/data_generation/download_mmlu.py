"""
Download MMLU from Hugging Face, convert to data_generation schema, save as pkl.
Required columns: question, subject, choices (list), answer (0-based int).
"""
import argparse
import os

import pandas as pd

# 57 cais/mmlu task subsets (excl. all / auxiliary_train). Fallback per-subset load if config=all fails.
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def letter_to_index(a):
    if isinstance(a, int) and 0 <= a <= 3:
        return a
    if isinstance(a, str):
        a = a.strip().upper()
        if a in ("A", "0"):
            return 0
        if a in ("B", "1"):
            return 1
        if a in ("C", "2"):
            return 2
        if a in ("D", "3"):
            return 3
    return None


def _load_dataset_kw(force_redownload: bool) -> dict:
    if force_redownload:
        return {"download_mode": "force_redownload"}
    return {}


def iter_mmlu_items(split: str, force_redownload: bool):
    """Try config=all first; on failure load 57 subsets (equivalent to full test)."""
    from datasets import load_dataset

    kw = _load_dataset_kw(force_redownload)
    try:
        ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True, **kw)
        for item in ds:
            yield item
        return
    except Exception as e:
        print(
            f"Warning: cannot load cais/mmlu with config=all ({type(e).__name__}: {e}).\n"
            "Loading 57 subsets one by one (equivalent to concatenating all).\n"
            "Optional fix: pip install -U 'datasets>=2.16' pyarrow; or clear cache and use --force_redownload:\n"
            "  rm -rf ~/.cache/huggingface/datasets/cais___mmlu*",
            flush=True,
        )
    for subj in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split=split, trust_remote_code=True, **kw)
        except Exception as e:
            print(f"  skip {subj} / {split}: {e}", flush=True)
            continue
        for item in ds:
            yield item


def main():
    parser = argparse.ArgumentParser(description="Download MMLU and save as raw_data/mmlu_raw.pkl")
    parser.add_argument(
        "--output",
        type=str,
        default="raw_data/mmlu_raw.pkl",
        help="Output pkl path; default raw_data/mmlu_raw.pkl (relative to cwd)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "dev", "auxiliary_train"],
        help="Dataset split; default test (~14k rows)",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Passed to datasets: force redownload; fixes corrupted local metadata cache",
    )
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Install first: pip install datasets")
        raise

    print(f"Loading cais/mmlu from Hugging Face (split={args.split}) ...")

    rows = []
    for item in iter_mmlu_items(args.split, args.force_redownload):
        question = item["question"]
        subject = item.get("subject")
        if not subject:
            subject = item.get("task") or ""
        choices = item["choices"]
        if isinstance(choices, str):
            import ast

            choices = ast.literal_eval(choices) if choices else []
        ans = item["answer"]
        if isinstance(ans, int) and 0 <= ans <= 3:
            answer_index = ans
        else:
            answer_index = letter_to_index(ans)
        if answer_index is None or not (0 <= answer_index < len(choices)):
            continue
        rows.append(
            {
                "question": question,
                "subject": subject,
                "choices": choices,
                "answer": answer_index,
            }
        )

    df = pd.DataFrame(rows)
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_pickle(out_path)
    print(f"Saved {len(df)} rows to {os.path.abspath(out_path)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
