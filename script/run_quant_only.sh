#!/usr/bin/env bash
# Quant only: appends --skip_eval --skip_plot (run python from repo root).
# Artifacts: cache_root/<model_id_fs>/<method_id>/fake_quant_model (method_id from --bits, e.g. rtn_w4).
#
# Example (no --eval; dataset key must exist in yaml syco.datasets):
#
#   python run_pipeline.py \
#     --dataset mmlu \
#     --model llama_3.1_8b_instruct \
#     --method RTN \
#     --bits 4 \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml \
#     --skip_eval \
#     --skip_plot
#
# Via this script (--skip_eval --skip_plot appended):
#
#   script/run_quant_only.sh \
#     --dataset mmlu \
#     --model llama_3.1_8b_instruct \
#     --method RTN \
#     --bits 4 \
#     --gpu cuda:0 \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py "$@" --skip_eval --skip_plot
