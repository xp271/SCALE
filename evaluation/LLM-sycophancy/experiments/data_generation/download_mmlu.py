"""
从 Hugging Face 下载 MMLU 数据集，并转换为 data_generation 所需格式后保存为 pkl。
所需列: question, subject, choices (list), answer (0-based int)。
"""
import argparse
import os

import pandas as pd

# cais/mmlu 的 57 个任务子集（不含 all / auxiliary_train）。当 config=all 因缓存/版本问题加载失败时按子集拼接。
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
    """优先加载 all；失败则按 57 子集拼接（与 test 全量一致）。"""
    from datasets import load_dataset

    kw = _load_dataset_kw(force_redownload)
    try:
        ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True, **kw)
        for item in ds:
            yield item
        return
    except Exception as e:
        print(
            f"警告: 无法以 config=all 加载 cais/mmlu（{type(e).__name__}: {e}）。\n"
            "改为按 57 个子集逐一加载（结果与 all 拼接等价）。\n"
            "可选修复: pip install -U 'datasets>=2.16' pyarrow；或清理缓存后加 --force_redownload：\n"
            "  rm -rf ~/.cache/huggingface/datasets/cais___mmlu*",
            flush=True,
        )
    for subj in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split=split, trust_remote_code=True, **kw)
        except Exception as e:
            print(f"  跳过 {subj} / {split}: {e}", flush=True)
            continue
        for item in ds:
            yield item


def main():
    parser = argparse.ArgumentParser(description="Download MMLU and save as raw_data/mmlu_raw.pkl")
    parser.add_argument(
        "--output",
        type=str,
        default="raw_data/mmlu_raw.pkl",
        help="输出 pkl 路径，默认 raw_data/mmlu_raw.pkl（相对当前工作目录）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "dev", "auxiliary_train"],
        help="使用的 split，默认 test（约 1.4 万条）",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="传给 datasets：强制重新下载，可修复损坏的本地元数据缓存",
    )
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("请先安装: pip install datasets")
        raise

    print(f"正在从 Hugging Face 加载 cais/mmlu (split={args.split}) ...")

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
    print(f"已保存 {len(df)} 条到 {os.path.abspath(out_path)}")
    print(f"列: {list(df.columns)}")


if __name__ == "__main__":
    main()
