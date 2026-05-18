#!/usr/bin/env bash
# Plot only: existing pkls under save_root/behavioral via run_pipeline -> figure.scan / figure.behavioral.
# Requires --plot_scan_existing and --plot_scan_model_id (optional --dataset, etc.).
#
# Example:
#
#   python run_pipeline.py \
#     --plot_scan_existing \
#     --plot_scan_model_id llama_3.1_8b_instruct \
#     --dataset mmlu \
#     --config config/pipeline_config.yaml
#
# This script adds --plot_scan_existing; other args pass through:
#
#   script/run_plot_only.sh \
#     --plot_scan_model_id llama_3.1_8b_instruct \
#     --dataset mmlu \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py --plot_scan_existing "$@"
