#!/usr/bin/env bash
# Eval only (skip plot): appends --skip_plot to the command.
# If fake_quant_model exists under cache_root, quant step is skipped automatically.
#
# Example:
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model llama_3.1_8b_instruct \
#     --method RTN \
#     --bits 4 \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml \
#     --skip_plot
#
# Via this script (--skip_plot appended):
#
#   script/run_eval_only.sh \
#     --dataset mmlu \
#     --model llama_3.1_8b_instruct \
#     --method RTN \
#     --bits 4 \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@" --skip_plot
