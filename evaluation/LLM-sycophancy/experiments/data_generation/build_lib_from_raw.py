"""
From raw pkl (MMLU / CommonsenseQA etc.) build experiment pkls under lib/:
- lib/plain/{data_slug}_plain_{seed}.pkl
- lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl
- lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{beginner,intermediate,advanced}_{seed}.pkl

Dependencies:
- raw pkl: from download_mmlu.py / download_commonsenseqa.py (columns question, subject, choices, answer)
- prefix pkl: generate_prefixes.py; filenames must match --output_name_prefix:
  {prefix_dir}/{output_name_prefix}_{level}_{seed}.pkl

Run from LLM-sycophancy repo root so lib/ lands under repo root.

CQA example:
  python experiments/data_generation/build_lib_from_raw.py \\
    --raw_file experiments/data_generation/raw_data/commonsenseqa_raw.pkl \\
    --output_name_prefix academic_prefix_commonsenseqa \\
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
DEFAULT_PREFIXMY_DIR = os.path.join(_REPO_ROOT, "prefixmy")


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


from full_question_builder import FullQuestionBuilder, match_category_prefix


def main():
    parser = argparse.ArgumentParser(
        description="Build plain, opinion_only, prefix+opinion (three levels) pkls under lib/ from raw pkl"
    )
    parser.add_argument(
        "--raw_file",
        type=str,
        default="raw_data/mmlu_raw.pkl",
        help="Raw pkl; resolve under experiments/data_generation first, else repo root",
    )
    parser.add_argument(
        "--prefix_dir",
        type=str,
        default=DEFAULT_PREFIXMY_DIR,
        help=f"Prefix pkl directory (default prefixmy under repo root: {DEFAULT_PREFIXMY_DIR})",
    )
    parser.add_argument(
        "--output_name_prefix",
        type=str,
        default="academic_prefix_mmlu",
        help="Must match generate_prefixes.py --output_name_prefix, e.g. academic_prefix_commonsenseqa for CQA",
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
        help="Random seed; must match generate_prefixes.py --seed when prefixes were built",
    )
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    raw_file = _resolve_under_data_gen(args.raw_file)
    prefix_dir = args.prefix_dir
    output_name_prefix = args.output_name_prefix
    data_slug = args.data_slug if args.data_slug else infer_data_slug(raw_file)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"raw data file not found: {raw_file}."
            f"Run download_mmlu.py or download_commonsenseqa.py first (or check --raw_file path)."
        )

    print(f"Reading raw data: {raw_file}")
    print(f"data_slug={data_slug}, output_name_prefix={output_name_prefix}, seed={seed}")
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

    # 1. Plain baseline
    print("Building plain baseline data ...")
    df_plain = builder.build_augmented(prefix_type="", question_style="plain")
    plain_path = f"lib/plain/{data_slug}_plain_{seed}.pkl"
    df_plain.to_pickle(plain_path)
    print(f"Wrote {plain_path}, samples: {len(df_plain)}")

    # 2. Opinion-only
    print("Building opinion-only data ...")
    df_opinion = builder.build_augmented(prefix_type="", question_style="opinion_only")
    opinion_path = f"lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl"
    df_opinion.to_pickle(opinion_path)
    print(f"Wrote {opinion_path}, samples: {len(df_opinion)}")

    # 3. POV + expertise (first-person)
    levels = ["beginner", "intermediate", "advanced"]
    for level in levels:
        prefix_file = os.path.join(
            prefix_dir, f"{output_name_prefix}_{level}_{seed}.pkl"
        )
        if not os.path.exists(prefix_file):
            raise FileNotFoundError(
                f"prefix file not found: {prefix_file}."
                f"Run first: python generate_prefixes.py --raw_file <same raw as here> "
                f"--output_name_prefix {output_name_prefix} --seed {seed} "
                f"(and --prefix_dir points to the same dir, default prefixmy under repo root)."
            )

        print(f"Building POV + expertise ({level}) data ...")
        prefix_df = pd.read_pickle(prefix_file)
        df_pov = builder.build_augmented(
            prefix_df=prefix_df,
            prefix_type="academic",
            prefix_selector_func=match_category_prefix,
            prefix_selector_args={"fallback_prefix": "I am an expert in this field."},
            question_style="prefix_and_opinion",
        )

        out_path = f"lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{level}_{seed}.pkl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_pov.to_pickle(out_path)
        print(f"Wrote {out_path}, samples: {len(df_pov)}")

    print("All lib/* pkls generated.")


if __name__ == "__main__":
    main()
