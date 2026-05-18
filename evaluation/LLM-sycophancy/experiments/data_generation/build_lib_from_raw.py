"""
从 raw pkl（MMLU / CommonsenseQA 等同 schema）生成 lib/ 目录下运行实验所需的 pkl 文件：
- lib/plain/{data_slug}_plain_{seed}.pkl
- lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl
- lib/pov/prefix/first_pov/{data_slug}_academic_opinion_{beginner,intermediate,advanced}_{seed}.pkl

依赖：
- raw pkl：download_mmlu.py / download_commonsenseqa.py 等生成（列 question, subject, choices, answer）
- 前缀 pkl：generate_prefixes.py，文件名须与 --output_name_prefix 一致：
  {prefix_dir}/{output_name_prefix}_{level}_{seed}.pkl

建议在 LLM-sycophancy 仓库根目录执行本脚本，使 lib/ 落在仓库根下。

CQA 示例：
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

# 确保可以找到同目录下的 full_question_builder.py
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

_REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_PREFIXMY_DIR = os.path.join(_REPO_ROOT, "prefixmy")


def _resolve_under_data_gen(path: str) -> str:
    """
    解析相对路径：先在 experiments/data_generation 下找，再在仓库根下找。
    这样既可写 raw_data/foo.pkl（相对 data_generation），也可写
    experiments/data_generation/raw_data/foo.pkl（相对仓库根，在仓库根执行时常见）。
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
    """与 full_question_builder 中 base_name 规则一致：basename 去扩展名并去掉 _raw 后缀。"""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    return stem if stem else "dataset"


from full_question_builder import FullQuestionBuilder, match_category_prefix


def main():
    parser = argparse.ArgumentParser(
        description="从 raw pkl 生成 lib/ 下的 plain、opinion_only、prefix+opinion（三档）pkl"
    )
    parser.add_argument(
        "--raw_file",
        type=str,
        default="raw_data/mmlu_raw.pkl",
        help="原始 pkl；相对路径时先在 experiments/data_generation 下解析，不存在则在仓库根下解析",
    )
    parser.add_argument(
        "--prefix_dir",
        type=str,
        default=DEFAULT_PREFIXMY_DIR,
        help=f"前缀 pkl 所在目录（默认仓库根下 prefixmy: {DEFAULT_PREFIXMY_DIR}）",
    )
    parser.add_argument(
        "--output_name_prefix",
        type=str,
        default="academic_prefix_mmlu",
        help="与 generate_prefixes.py --output_name_prefix 一致，例如 CQA 用 academic_prefix_commonsenseqa",
    )
    parser.add_argument(
        "--data_slug",
        type=str,
        default=None,
        help="写入 lib/ 的文件名前缀；默认从 --raw_file 推断（basename 去掉 _raw 与扩展名）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子；须与生成前缀时 generate_prefixes.py 的 --seed 一致",
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
            f"找不到原始数据文件 {raw_file}。"
            f"请先运行 download_mmlu.py 或 download_commonsenseqa.py（或检查 --raw_file 路径）。"
        )

    print(f"读取原始数据：{raw_file}")
    print(f"data_slug={data_slug}, output_name_prefix={output_name_prefix}, seed={seed}")
    df_raw = pd.read_pickle(raw_file)

    # 与 data_generation 代码保持一致的列映射
    column_mapping = {
        "question": "question",
        "subject": "category",
        "choices": "options",
        "answer": "answer_index",
    }

    builder = FullQuestionBuilder(
        df_raw,
        base_output_dir=".",  # 输出目录对我们没影响，只用它的 build_augmented
        column_mapping=column_mapping,
    )

    # 确保 lib 目录结构存在
    paths_to_create = [
        "lib/plain",
        "lib/opinion_only/prefix",
        "lib/pov/prefix/first_pov",
    ]
    for p in paths_to_create:
        os.makedirs(p, exist_ok=True)

    # 1. Plain baseline
    print("生成 plain 基线数据 ...")
    df_plain = builder.build_augmented(prefix_type="", question_style="plain")
    plain_path = f"lib/plain/{data_slug}_plain_{seed}.pkl"
    df_plain.to_pickle(plain_path)
    print(f"已写入 {plain_path}，样本数：{len(df_plain)}")

    # 2. Opinion-only
    print("生成 opinion-only 数据 ...")
    df_opinion = builder.build_augmented(prefix_type="", question_style="opinion_only")
    opinion_path = f"lib/opinion_only/prefix/{data_slug}_opinion_only_{seed}.pkl"
    df_opinion.to_pickle(opinion_path)
    print(f"已写入 {opinion_path}，样本数：{len(df_opinion)}")

    # 3. POV + expertise (first-person)
    levels = ["beginner", "intermediate", "advanced"]
    for level in levels:
        prefix_file = os.path.join(
            prefix_dir, f"{output_name_prefix}_{level}_{seed}.pkl"
        )
        if not os.path.exists(prefix_file):
            raise FileNotFoundError(
                f"找不到前缀文件 {prefix_file}。"
                f"请先运行：python generate_prefixes.py --raw_file <与当前相同的 raw> "
                f"--output_name_prefix {output_name_prefix} --seed {seed} "
                f"(且 --prefix_dir 指向同一目录，默认即仓库根下 prefixmy)。"
            )

        print(f"生成 POV + expertise ({level}) 数据 ...")
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
        print(f"已写入 {out_path}，样本数：{len(df_pov)}")

    print("全部 lib/* pkl 生成完毕。")


if __name__ == "__main__":
    main()
