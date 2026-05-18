#!/usr/bin/env bash
# 只绘图：基于 save_root/behavioral 下已有 pkl，经 run_pipeline → figure.scan / figure.behavioral。
# 须带上 --plot_scan_existing 与 --plot_scan_model_id（以及可选 --dataset 等）。
#
# 示例：
#
#   python run_pipeline.py \
#     --plot_scan_existing \
#     --plot_scan_model_id mistral_7b_instruct_v0_3 \
#     --dataset mmlu \
#     --config config/pipeline_config.yaml
#
# 通过本脚本自动补上 --plot_scan_existing，其余参数原样透传：
#
#   script/run_plot_only.sh \
#     --plot_scan_model_id mistral_7b_instruct_v0_3 \
#     --dataset mmlu \
#     --config config/pipeline_config.yaml
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
exec python run_pipeline.py --plot_scan_existing "$@"
