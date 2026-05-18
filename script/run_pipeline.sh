#!/usr/bin/env bash
# Full pipeline: quantize -> eval -> plot.
# Runs python from repo root; this script cd's to root first.
#
# Example (adjust args as needed):
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model llama_3.1_8b_instruct \
#     --method RTN \
#     --bits 4 \
#     --eval behavioral,mechanistic \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
# Equivalent via this script (args passed through):
#
#   script/run_pipeline.sh \
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
exec python run_pipeline.py "$@"
