#!/usr/bin/env bash
# 只跑量化：在命令末尾追加 --skip_eval --skip_plot（须在仓库根执行 python）。
# 量化产物落在 cache_root/<model_id_fs>/<method_id>/fake_quant_model（method_id 由 --bits 决定，如 awq_w4）。
#
# 示例（无 --eval；数据集键仍需存在于 yaml 的 syco.datasets）：
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model mistral_7b_instruct_v0_3 \
#     --method Awq \
#     --bits 4 \
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
#     --bits 4 \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@" --skip_eval --skip_plot
