"""CLI parser + yaml validation/narrowing for run_pipeline.py.

The pipeline is driven by required runtime parameters (``--dataset``,
``--model``, ``--method``, ``--bits``, ``--eval``, ``--gpu``; ``--eval`` may be
omitted with ``--skip_eval``). Everything else lives in
``config/pipeline_config.yaml``. This module:

- Parses argv with :func:`parse_cli` into a :class:`CliArgs` dataclass.
- Validates that the requested ``--dataset / --model / --method`` exist in
  the yaml and narrows the config to those single entries via
  :func:`narrow_config_for_cli`.
- Expands the comma list ``--eval`` into fine-grained job tags via
  :func:`expand_eval_modes`.
- Normalizes ``--gpu`` strings like ``cuda:0`` / ``0,1`` into
  ``CUDA_VISIBLE_DEVICES`` values via :func:`normalize_gpu`.
- Recognizes the ``--plot_scan_existing`` short-circuit mode (no 5-required).
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.seeds import generate_eval_data_seeds

# --- Eval mode expansion ----------------------------------------------------

_BEHAVIORAL_TOP = "behavioral"
_MECHANISTIC_TOP = "mechanistic"
_LOGIT_COT_ALIAS = "logit_cot"

# 顶层 token → 展开后的细粒度 tag 集合
_TOP_LEVEL_EXPANSION: dict[str, tuple[str, ...]] = {
    _BEHAVIORAL_TOP: ("plain", "opinion_only"),
    _MECHANISTIC_TOP: ("logit_cot_plain", "logit_cot_opinion"),
    _LOGIT_COT_ALIAS: ("logit_cot_plain", "logit_cot_opinion"),
}

# CLI 允许的细粒度 tag（job_builder 内部 tag 或 authority 这种聚合开关）
_FINE_GRAINED: frozenset[str] = frozenset(
    {
        "plain",
        "opinion_only",
        "authority",
        "behavior_prefix",
        "logit_cot_plain",
        "logit_cot_opinion",
    }
)

# 仅影响绘图/统计口径（eval_sr_correct_only），不参与 job tag 过滤
_EVAL_META_TAGS: frozenset[str] = frozenset({"correct_only", "full_sr"})
_CORRECT_ONLY_TAG = "correct_only"
_FULL_SR_TAG = "full_sr"

# 用于推导 yaml 的三个布尔
_AUTHORITY_TAG = "authority"
_BEHAVIOR_PREFIX_TAG = "behavior_prefix"
_MECHANISTIC_TAGS: frozenset[str] = frozenset({"logit_cot_plain", "logit_cot_opinion"})


@dataclass(frozen=True)
class CliArgs:
    """Parsed CLI arguments.

    Non plot-only invocations require ``dataset / model_id / method / gpu``
    plus ``weight_bits`` (``--bits``), and ``eval_modes`` unless ``--skip_eval``.
    plot-only invocations may leave the base args blank and only need
    ``plot_scan_model_id`` (``plot_scan_existing=True``).
    """

    dataset: str | None
    model_id: str | None
    method: str | None
    weight_bits: int | None
    eval_modes: tuple[str, ...]
    gpu: str | None
    config_path: Path
    # 阶段跳过开关
    skip_eval: bool
    skip_plot: bool
    # plot-only 分支
    plot_scan_existing: bool
    plot_scan_model_id: str | None
    plot_scan_seeds: str | None
    plot_scan_figure_dir: str | None
    plot_scan_correct_only: str | None
    # 评估：多 seed 数据准备与绘图平均（--plot_scan_seeds 未给时用同一套）
    eval_avg_runs: int
    data_seed_rng: int


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / "pipeline_config.yaml"


def parse_cli(argv: list[str] | None = None) -> CliArgs:
    """Parse CLI argv into :class:`CliArgs`. Exits with usage error on misuse."""
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Run one (dataset, model, method, --bits, eval set) combo on a single GPU. "
            "Use --plot_scan_existing to re-plot from existing pkls without "
            "running quantization or evaluation."
        ),
    )
    p.add_argument("--dataset", help="Dataset id; must exist in yaml syco.datasets")
    p.add_argument("--model", dest="model_id", help="Model id; must exist in yaml models[*].model_id")
    p.add_argument("--method", help="Quant method name; must exist in yaml methods[*].method (e.g. Awq)")
    p.add_argument(
        "--bits",
        dest="weight_bits",
        type=int,
        metavar="N",
        help="Weight quantization width (e.g. 4 for W4). Required unless --plot_scan_existing.",
    )
    p.add_argument(
        "--eval",
        dest="eval_modes",
        help=(
            "Comma list of eval modes. Top-level: behavioral | mechanistic | logit_cot. "
            "Fine-grained: plain | opinion_only | authority | behavior_prefix | "
            "logit_cot_plain | logit_cot_opinion. "
            "Meta (SR scope): correct_only (opt-in correct-only SR for behavioral figs; default is full SR). "
            "full_sr forces full SR when yaml had eval_sr_correct_only true (mutually exclusive with correct_only). "
            "Multi-select union."
        ),
    )
    p.add_argument(
        "--gpu",
        help="GPU spec. Accepts 'cuda:N', 'N', or 'N,M'. Maps to CUDA_VISIBLE_DEVICES; syco device becomes cuda:0.",
    )
    p.add_argument(
        "--eval_avg_runs",
        type=int,
        default=3,
        metavar="N",
        help=(
            "评估与绘图平均用的数据 seed 个数。与 --data_seed_rng 共同决定 syco.data_seed 列表 "
            "（仅在未 --skip_eval 时写入配置；plot-only 且无 --plot_scan_seeds 时用于选 seed）。默认 3。"
        ),
    )
    p.add_argument(
        "--data_seed_rng",
        type=int,
        default=42,
        metavar="S",
        help="用于生成上述 N 个 data_seed 的随机数发生器种子（可复现）。默认 42。",
    )
    p.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help=f"Pipeline yaml (default: {_default_config_path()})",
    )

    # 阶段跳过开关：quant-only / eval-only 用
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip syco eval. Useful for quant-only runs (--eval may be omitted).",
    )
    p.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip the plot phase. Useful for quant-only or eval-only runs.",
    )

    # plot-only 短路分支
    p.add_argument("--plot_scan_existing", action="store_true", help="Plot from existing pkls; skip quant/eval.")
    p.add_argument("--plot_scan_model_id", default=None, help="(plot-only) model id whose pkls to scan")
    p.add_argument("--plot_scan_seeds", default=None, help="(plot-only) explicit comma-separated data_seeds; if omitted uses --data_seed_rng + --eval_avg_runs")
    p.add_argument("--plot_scan_figure_dir", default=None, help="(plot-only) override figure output dir")
    p.add_argument("--plot_scan_correct_only", default=None, help="(plot-only) override correct_only_sr flag")

    ns = p.parse_args(argv)

    config_path = Path(ns.config_path).resolve() if ns.config_path else _default_config_path()

    if ns.plot_scan_existing:
        if not ns.plot_scan_model_id:
            p.error("--plot_scan_existing requires --plot_scan_model_id <model_id>")
        if ns.eval_avg_runs < 1:
            p.error("--eval_avg_runs must be >= 1")
        return CliArgs(
            dataset=ns.dataset,
            model_id=None,
            method=None,
            weight_bits=None,
            eval_modes=tuple(),
            gpu=None,
            config_path=config_path,
            skip_eval=True,
            skip_plot=False,
            plot_scan_existing=True,
            plot_scan_model_id=ns.plot_scan_model_id,
            plot_scan_seeds=ns.plot_scan_seeds,
            plot_scan_figure_dir=ns.plot_scan_figure_dir,
            plot_scan_correct_only=ns.plot_scan_correct_only,
            eval_avg_runs=ns.eval_avg_runs,
            data_seed_rng=ns.data_seed_rng,
        )

    # 普通运行模式：4 必填 (dataset / model / method / gpu) + bits + 1 条件必填 (--eval)
    base_required = (
        ("--dataset", ns.dataset),
        ("--model", ns.model_id),
        ("--method", ns.method),
        ("--gpu", ns.gpu),
    )
    missing = [name for name, val in base_required if not val]
    if ns.weight_bits is None:
        missing.append("--bits")
    elif ns.weight_bits <= 0:
        p.error("--bits must be a positive integer")
    if ns.eval_avg_runs < 1:
        p.error("--eval_avg_runs must be >= 1")
    if not ns.skip_eval and not ns.eval_modes:
        missing.append("--eval")
    if missing:
        p.error(f"missing required arguments: {', '.join(missing)}")

    raw_modes: list[str] = []
    if ns.eval_modes:
        raw_modes = [t.strip() for t in str(ns.eval_modes).split(",") if t.strip()]
        valid_tokens = set(_TOP_LEVEL_EXPANSION) | _FINE_GRAINED | _EVAL_META_TAGS
        bad = [t for t in raw_modes if t not in valid_tokens]
        if bad:
            p.error(
                f"--eval got unknown token(s): {bad}. "
                f"Valid: {sorted(valid_tokens)}"
            )
        if _CORRECT_ONLY_TAG in raw_modes and _FULL_SR_TAG in raw_modes:
            p.error("--eval cannot combine correct_only with full_sr (choose one).")

    try:
        normalize_gpu(ns.gpu)
    except ValueError as e:
        p.error(str(e))

    return CliArgs(
        dataset=ns.dataset,
        model_id=ns.model_id,
        method=ns.method,
        weight_bits=ns.weight_bits,
        eval_modes=tuple(raw_modes),
        gpu=ns.gpu,
        config_path=config_path,
        skip_eval=bool(ns.skip_eval),
        skip_plot=bool(ns.skip_plot),
        plot_scan_existing=False,
        plot_scan_model_id=None,
        plot_scan_seeds=None,
        plot_scan_figure_dir=None,
        plot_scan_correct_only=None,
        eval_avg_runs=ns.eval_avg_runs,
        data_seed_rng=ns.data_seed_rng,
    )


def expand_eval_modes(tokens: Iterable[str]) -> set[str]:
    """Expand a list of CLI eval tokens into fine-grained tags.

    ``behavioral`` → ``{plain, opinion_only}``; ``mechanistic`` / ``logit_cot``
    → ``{logit_cot_plain, logit_cot_opinion}``. Fine-grained tokens are kept
    as-is. ``authority`` and ``behavior_prefix`` map to the same string and
    are interpreted by callers (they're agg switches, not job tags).
    Meta tokens ``correct_only`` / ``full_sr`` are kept for :func:`narrow_config_for_cli`
    and stripped by :func:`eval_tags_for_jobs` before job filtering.
    """
    out: set[str] = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in _TOP_LEVEL_EXPANSION:
            out.update(_TOP_LEVEL_EXPANSION[tok])
        elif tok in _FINE_GRAINED:
            out.add(tok)
        elif tok in _EVAL_META_TAGS:
            out.add(tok)
        else:
            raise ValueError(f"unknown eval token: {tok}")
    return out


def eval_tags_for_jobs(tags: set[str]) -> set[str]:
    """Subset tags used by :func:`filter_eval_jobs_by_tags` (drop SR meta tokens)."""
    return tags - _EVAL_META_TAGS


def eval_flags_from_tags(tags: set[str]) -> dict[str, bool]:
    """Derive yaml-equivalent boolean knobs from expanded eval tags."""
    return {
        "eval_mechanistic": bool(tags & _MECHANISTIC_TAGS),
        "eval_authority_advanced": _AUTHORITY_TAG in tags,
        "eval_behavior_prefix": _BEHAVIOR_PREFIX_TAG in tags,
    }


_CUDA_PREFIX_RE = re.compile(r"^cuda:(\d+(?:,\d+)*)$", re.IGNORECASE)


def normalize_gpu(gpu: str | None) -> str | None:
    """Normalize ``--gpu`` to a ``CUDA_VISIBLE_DEVICES`` value (or None to leave unset).

    Accepts ``cuda:0``, ``cuda:0,1``, ``0``, ``0,1``. ``auto`` / ``all`` /
    ``none`` / ``default`` (any case) return ``None`` so the env var is not
    narrowed.
    """
    if gpu is None:
        return None
    s = gpu.strip()
    if not s:
        return None
    if s.lower() in ("auto", "all", "none", "default"):
        return None
    m = _CUDA_PREFIX_RE.match(s)
    if m:
        return m.group(1)
    if all(part.strip().isdigit() for part in s.split(",") if part.strip()):
        return s
    raise ValueError(f"--gpu got unrecognized value: {gpu!r} (expected 'cuda:N', 'N', or 'N,M')")


# --- yaml validation / narrowing -------------------------------------------


def _format_known(items: list[str]) -> str:
    return ", ".join(repr(i) for i in items) if items else "<empty>"


def _validate_model(cfg: dict, model_id: str, config_path: Path) -> dict:
    models = cfg.get("models") or []
    matches = [m for m in models if m.get("model_id") == model_id]
    if not matches:
        known = [m.get("model_id") for m in models if m.get("model_id")]
        print(
            f"--model '{model_id}' 未在 {config_path} 的 models 列表中。\n"
            f"已配置: [{_format_known(known)}]。请在 yaml 中补一条 models 条目。",
            file=sys.stderr,
        )
        sys.exit(2)
    return matches[0]


def _validate_method(cfg: dict, method: str, config_path: Path) -> dict:
    methods = cfg.get("methods") or []
    matches = [m for m in methods if m.get("method") == method]
    if not matches:
        known = [m.get("method") for m in methods if m.get("method")]
        print(
            f"--method '{method}' 未在 {config_path} 的 methods 列表中。\n"
            f"已配置: [{_format_known(known)}]。请在 yaml 中补一条 methods 条目。",
            file=sys.stderr,
        )
        sys.exit(2)
    return matches[0]


def _validate_dataset(cfg: dict, dataset: str, config_path: Path) -> dict:
    syco_cfg = cfg.get("syco") or {}
    datasets = syco_cfg.get("datasets") or {}
    if dataset not in datasets:
        print(
            f"--dataset '{dataset}' 未在 {config_path} 的 syco.datasets 中。\n"
            f"已配置: [{_format_known(sorted(datasets.keys()))}]。请在 yaml 中补一条 syco.datasets 条目。",
            file=sys.stderr,
        )
        sys.exit(2)
    return datasets[dataset]


def narrow_config_for_cli(cfg: dict, cli: CliArgs) -> dict:
    """Return a deep-copied cfg narrowed to CLI selections.

    - ``models`` / ``methods`` are reduced to the single matching entry.
    - ``syco.datasets[--dataset]`` is merged onto ``syco`` (overriding
      ``data_slug / raw_file / download_script / build_lib_extra_args``).
    - ``syco.dataset`` is set to the CLI value.
    - ``syco.eval_mechanistic / eval_authority_advanced / eval_behavior_prefix``
      booleans are overwritten from CLI tags.
    - ``syco.eval_sr_correct_only`` is set from meta tags ``correct_only`` / ``full_sr``
      when present; otherwise the yaml value is kept.
    - When not ``skip_eval``, ``syco.data_seed`` is replaced by
      :func:`utils.seeds.generate_eval_data_seeds` from ``cli.data_seed_rng`` and
      ``cli.eval_avg_runs`` (yaml 中的列表不再使用）。
    - Top-level ``cuda_device`` is overwritten with :func:`normalize_gpu`.

    Validation failures call :func:`sys.exit(2)` with a friendly message.
    """
    cfg = copy.deepcopy(cfg)
    config_path = cli.config_path

    if cli.plot_scan_existing:
        # plot-only：只在 dataset 给出时校验，否则随后由调用方决定取哪个 dataset
        if cli.dataset:
            _validate_dataset(cfg, cli.dataset, config_path)
        return cfg

    assert cli.dataset and cli.model_id and cli.method  # parse_cli 已经保证

    model_entry = _validate_model(cfg, cli.model_id, config_path)
    method_entry = _validate_method(cfg, cli.method, config_path)
    dataset_entry = _validate_dataset(cfg, cli.dataset, config_path)

    cfg["models"] = [model_entry]
    cfg["methods"] = [method_entry]

    syco = cfg.setdefault("syco", {})
    syco["dataset"] = cli.dataset
    # merge dataset-specific knobs onto syco top-level
    for k, v in (dataset_entry or {}).items():
        syco[k] = v

    tags = expand_eval_modes(cli.eval_modes)
    syco.update(eval_flags_from_tags(tags))
    if _CORRECT_ONLY_TAG in tags:
        syco["eval_sr_correct_only"] = True
    elif _FULL_SR_TAG in tags:
        syco["eval_sr_correct_only"] = False

    if not cli.skip_eval:
        syco["data_seed"] = generate_eval_data_seeds(cli.data_seed_rng, cli.eval_avg_runs)

    cfg["cuda_device"] = normalize_gpu(cli.gpu)
    return cfg


def filter_eval_jobs_by_tags(eval_jobs: list[dict], tags: set[str]) -> list[dict]:
    """Filter a job list (as returned by ``build_syco_eval_jobs``) by CLI tags.

    ``authority`` is special: it represents the three ``authority_*`` jobs in
    job_builder (beginner / intermediate / advanced). When present in CLI tags,
    we let any ``authority_*`` tag through.
    """
    authority_on = _AUTHORITY_TAG in tags
    return [
        j
        for j in eval_jobs
        if j["tag"] in tags or (authority_on and str(j["tag"]).startswith("authority_"))
    ]
