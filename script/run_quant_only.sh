#!/usr/bin/env bash
# 只跑量化：在命令末尾追加 --skip_eval --skip_plot（须在仓库根执行 python）。
# 量化产物落在 cache_root/mistral_7b_instruct_v0_3/awq_w{4,6,8}/fake_quant_model 等路径。
#
# 示例（无 --eval；数据集键仍需存在于 yaml 的 syco.datasets）：
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml \
#     --skip_eval \
#     --skip_plot
#
# 通过本脚本（自动补上最后两行）：
#
#   script/run_quant_only.sh \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@" --skip_eval --skip_plot
