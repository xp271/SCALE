#!/usr/bin/env bash
# 全流程：量化 -> 评估 -> 绘图。
# 在项目根目录执行 python；本脚本会先 cd 到仓库根再调用。
#
# 示例（与其它参数写法一致，按需删改）：
#
  python run_pipeline.py \
    --dataset commonsenseqa \
    --model llama_3.1_8b_instruct \
    --method RTN \
    --bits 4 \
    --eval behavioral \
    --gpu cuda:0 
#
# 通过本脚本等价调用（参数原样透传）：
#
#   script/run_pipeline.sh \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --bits 4 \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@"
