"""
From raw pkl (MMLU / CommonsenseQA etc.) build experiment pkls under lib/:
- lib/plain/{data_slug}_plain_{seed}.pkl
- lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl
- lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{beginner,intermediate,advanced}_{seed}.pkl

Dependencies:
- raw pkl: from download_mmlu.py / download_commonsenseqa.py (columns question, subject, choices, answer)

Prompts follow the paper (Section 3 "Prompts", Table 4):
- The user's belief ("I believe the answer is X. ...") is one incorrect option drawn once per question
  and shared by the Opinion-only prompt and all three Opinion-with-Authority prompts.
- Opinion-with-Authority prepends the original persona of each level (Novice / Competent / Expert,
  see prompt_templates.py), e.g. "As a Novice in {subject}, I'm just starting to learn ...".

Run from LLM-sycophancy repo root so lib/ lands under repo root.

CQA example:
  python experiments/data_generation/build_lib_from_raw.py \\
    --raw_file experiments/data_generation/raw_data/commonsenseqa_raw.pkl \\
    --seed 42
"""

import os
import sys
import argparse
import random
import pandas as pd

# Ensure full_question_builder.py in same dir is importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

_REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _resolve_under_data_gen(path: str) -> str:
    """
    Resolve relative path: try experiments/data_generation first, then repo root.
    Supports raw_data/foo.pkl (relative to data_generation) or
    experiments/data_generation/raw_data/foo.pkl (repo-relative when run from root).
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    here_p = os.path.abspath(os.path.join(HERE, path))
    repo_p = os.path.abspath(os.path.join(_REPO_ROOT, path))
    if os.path.isfile(here_p):
        return here_p
    if os.path.isfile(repo_p):
        return repo_p
    return here_p


def infer_data_slug(raw_path: str) -> str:
    """Same base_name rule as full_question_builder: basename without extension and _raw suffix."""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    return stem if stem else "dataset"


from full_question_builder import FullQuestionBuilder
from prompt_templates import AUTHORITY_LEVELS, ORIGINAL_PERSONA, build_authority_prefix


def main():
    parser = argparse.ArgumentParser(
        description="Build plain, opinion_only, opinion-with-authority (three levels) pkls under lib/ from raw pkl"
    )
    parser.add_argument(
        "--raw_file",
        type=str,
        default="raw_data/mmlu_raw.pkl",
        help="Raw pkl; resolve under experiments/data_generation first, else repo root",
    )
    parser.add_argument(
        "--data_slug",
        type=str,
        default=None,
        help="Filename prefix under lib/; default inferred from --raw_file (strip _raw and extension)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the user's (incorrect) belief drawn per question",
    )
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    raw_file = _resolve_under_data_gen(args.raw_file)
    data_slug = args.data_slug if args.data_slug else infer_data_slug(raw_file)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"raw data file not found: {raw_file}."
            f"Run download_mmlu.py or download_commonsenseqa.py first (or check --raw_file path)."
        )

    print(f"Reading raw data: {raw_file}")
    print(f"data_slug={data_slug}, seed={seed}")
    df_raw = pd.read_pickle(raw_file)

    # Column mapping consistent with data_generation code
    column_mapping = {
        "question": "question",
        "subject": "category",
        "choices": "options",
        "answer": "answer_index",
    }

    builder = FullQuestionBuilder(
        df_raw,
        base_output_dir=".",  # output dir unused; only build_augmented is used
        column_mapping=column_mapping,
    )

    # Ensure lib directory layout exists
    paths_to_create = [
        "lib/plain",
        "lib/opinion_only/prefix",
        "lib/pov/prefix/first_pov",
    ]
    for p in paths_to_create:
        os.makedirs(p, exist_ok=True)

    # 1. Plain baseline (consumes no randomness)
    print("Building plain baseline data ...")
    df_plain = builder.build_augmented(prefix_type="", question_style="plain")
    plain_path = f"lib/plain/{data_slug}_plain_{seed}.pkl"
    df_plain.to_pickle(plain_path)
    print(f"Wrote {plain_path}, samples: {len(df_plain)}")

    # 2. Opinion-only. The wrong option is drawn once here and reused by the authority prompts.
    print("Building opinion-only data ...")
    builder.fix_wrong_answers()
    df_opinion = builder.build_augmented(prefix_type="", question_style="opinion_only")
    opinion_path = f"lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl"
    df_opinion.to_pickle(opinion_path)
    print(f"Wrote {opinion_path}, samples: {len(df_opinion)}")

    # 3. Opinion-with-Authority: original persona of each level + the same user opinion
    for level in AUTHORITY_LEVELS:
        print(f"Building opinion-with-authority ({level}) data ...")
        df_pov = builder.build_augmented(
            prefix_type="academic",
            prefix_func=lambda subject, _level=level: build_authority_prefix(_level, subject, ORIGINAL_PERSONA),
            question_style="prefix_and_opinion",
        )
        if not df_pov["chosen_wrong_answer_index"].equals(df_opinion["chosen_wrong_answer_index"]):
            raise RuntimeError(f"{level}: user belief differs from the opinion-only data")

        out_path = f"lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{level}_{seed}.pkl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_pov.to_pickle(out_path)
        print(f"Wrote {out_path}, samples: {len(df_pov)}")

    print("All lib/* pkls generated.")


if __name__ == "__main__":
    main()
