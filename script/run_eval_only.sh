#!/usr/bin/env bash
# 只跑评估（并跳过绘图）：在命令末尾追加 --skip_plot。
# 若 cache_root 下对应 fake_quant_model 已存在，量化阶段会自动跳过。
#
# 示例：
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml \
#     --skip_plot
#
# 通过本脚本（自动补上最后一行）：
#
#   script/run_eval_only.sh \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@" --skip_plot
