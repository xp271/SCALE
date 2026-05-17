
<div align="center">

## eva_syc · Quantization × Sycophancy Pipeline

LightCompress（fake quant）与 LLM-sycophancy 行为/机理评估的一站式编排入口。

</div>

---

## Overview

本仓库在单次运行中串联：

1. **量化**：按配置调用 LightCompress，产出 `fake_quant_model`（写入 `cache_root`）。
2. **评估**：按 CLI 指定的数据集、模型、方法与评估子任务，调用 LLM-sycophancy 下的 `run_syco.py` / `run_syco_logit_cot.py`，输出写入 `save_root`（behavioral / mechanistic 子目录）。
3. **制图**（可选）：调用上游绘图脚本，产出写入 `result_root`。

运行时 **五个核心参数** 通过命令行传入：`--dataset`、`--model`、`--method`、`--eval`、`--gpu`；其余默认写在 [`config/pipeline_config.yaml`](config/pipeline_config.yaml)。入口脚本为根目录下的 [`run_pipeline.py`](run_pipeline.py)。

---

## Repository Structure

```
.
├── config/
│   ├── pipeline_config.yaml    # 默认配置（路径根、模型池、方法池、syco.datasets 等）
│   └── parser.py               # CLI 解析与 yaml 校验、窄化
├── quantization/
│   ├── runner.py               # LightCompress torchrun
│   ├── yml_builder.py
│   ├── registry.py
│   └── methods/                # 各量化方法（Awq、GPTQ、…）
├── evaluation/
│   ├── runner.py               # subprocess 调度 / eval_model_against_jobs
│   ├── job_builder.py
│   ├── paths.py
│   └── ...
├── figure/
│   ├── orchestrator.py
│   └── ...
├── utils/
│   ├── paths.py                # cache/save/result 根解析
│   └── ...
├── script/
│   ├── run_pipeline.sh         # cd 到仓库根后透传参数给 run_pipeline.py
│   ├── run_quant_only.sh       # 末尾追加 --skip_eval --skip_plot
│   ├── run_eval_only.sh        # 末尾追加 --skip_plot
│   └── run_plot_only.sh        # 前缀 --plot_scan_existing
├── cache/                      # 默认：HF 缓存 + 量化产物（见 yaml cache_root）
├── save/                       # 默认：syco pkl（见 yaml save_root）
├── result/                     # 默认：图（见 yaml result_root）
├── LightCompress/              # 外部仓库：需在 yaml llmc_dir 或默认路径可访问
├── LLM-sycophancy/             # 外部仓库：需在 yaml syco_repo_dir 或默认路径可访问
├── run_pipeline.py             # 推荐入口（CLI 驱动）
├── run_quant_syco_pipeline.py  # 历史单体脚本（仍可单独使用）
├── README_pipeline.md          # 旧版 pipeline 说明（面向 run_quant_syco_pipeline.py）
└── README.md                   # 本文件
```

---

## Quick Start

> **Note:** `LightCompress` 与 `LLM-sycophancy` 须能被 [`config/pipeline_config.yaml`](config/pipeline_config.yaml) 中的 `llmc_dir` / `syco_repo_dir` 解析到（默认可指向仓库内 `../quantization/LightCompress` 与 `../evaluation/LLM-sycophancy`，若你放在根目录同名文件夹则需改 yaml）。

### Installation

```bash
cd /path/to/eva_syc

# Python 依赖：请按 LightCompress、LLM-sycophancy 各自 requirements 安装
# 示例（按需调整）
pip install -r LightCompress/requirements.txt
pip install -r LLM-sycophancy/requirements.txt
pip install pyyaml
```

### Running the Pipeline

在项目根目录执行（或使用 `script/run_pipeline.sh`，会先 `cd` 到仓库根再调用 Python）。

#### 全流程（量化 + 评估 + 绘图）

```bash
python run_pipeline.py \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --eval behavioral,mechanistic \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml
```

#### 仅量化

```bash
python run_pipeline.py \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml \
  --skip_eval \
  --skip_plot
```

#### 仅评估（fake_quant 已存在时将跳过量化）

```bash
python run_pipeline.py \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --eval behavioral,mechanistic \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml \
  --skip_plot
```

#### 仅扫描已有 pkl 绘图

```bash
python run_pipeline.py \
  --plot_scan_existing \
  --plot_scan_model_id mistral_7b_instruct_v0_3 \
  --dataset mmlu \
  --config config/pipeline_config.yaml
```

### Shell Wrappers

脚本会先切换到仓库根目录，再把参数传给 `run_pipeline.py`（等价于你在根目录手动敲 `python run_pipeline.py …`）。

```bash
script/run_pipeline.sh \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --eval behavioral,mechanistic \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml
```

```bash
script/run_quant_only.sh \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml
```

```bash
script/run_eval_only.sh \
  --dataset mmlu \
  --model mistral_7b_instruct_v0_3 \
  --method Awq \
  --eval behavioral,mechanistic \
  --gpu cuda:0 \
  --config config/pipeline_config.yaml
```

```bash
script/run_plot_only.sh \
  --plot_scan_model_id mistral_7b_instruct_v0_3 \
  --dataset mmlu \
  --config config/pipeline_config.yaml
```

---

## CLI Reference

| 参数 | 说明 |
|------|------|
| `--dataset` | 须在 yaml `syco.datasets` 中有条目（如 `mmlu`、`commonsenseqa`）。 |
| `--model` | 须在 yaml `models[*].model_id` 中已有定义。 |
| `--method` | 须在 yaml `methods[*].method` 中已有定义（如 `Awq`）；默认对该方法跑 4/6/8 bit。 |
| `--eval` | 逗号列表：顶层 `behavioral`、`mechanistic`、`logit_cot`，或细粒度 `plain`、`opinion_only`、`authority`、`behavior_prefix`、`logit_cot_plain`、`logit_cot_opinion`。 |
| `--gpu` | `cuda:N`、`N` 或 `N,M`，映射为 `CUDA_VISIBLE_DEVICES`；子进程内推理设备为 `cuda:0`。 |
| `--config` | 配置文件路径；默认 `config/pipeline_config.yaml`。 |
| `--skip_eval` | 跳过全部 syco 评估与数据准备。 |
| `--skip_plot` | 跳过制图阶段。 |
| `--plot_scan_existing` | 仅绘图模式；需配合 `--plot_scan_model_id` 等。 |

未列出的训练种子、`plot_figures`、`aggregate_accuracy_csv`、校准/评测自动下载等见 yaml。

---

## Configuration

[`config/pipeline_config.yaml`](config/pipeline_config.yaml) 中的主要内容：

- **路径**：`cache_root`、`save_root`、`result_root`；`llmc_dir`、`syco_repo_dir`。
- **模型与方法候选池**：`models`、`methods`（CLI 只从中选取一项，缺失会报错提示补全）。
- **`syco.datasets`**：按 `--dataset` 键合并 `data_slug`、`raw_file`、`download_script`、`build_lib_extra_args`。
- **其余 `syco` 字段**：如 `data_seed`、`eval_sr_correct_only`、`max_retries` 等，不按 CLI 暴露时写在 yaml 内。

---

## Related Documentation

- 上游评估仓库结构与单独跑实验的方式，见 [LLM-sycophancy/README.md](LLM-sycophancy/README.md)。
- 面向旧入口 `run_quant_syco_pipeline.py` 的配置说明见 [README_pipeline.md](README_pipeline.md)。
