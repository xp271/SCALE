"""Build the per-task LightCompress run.yml from a method's template."""
from __future__ import annotations

from pathlib import Path

import yaml

# Placeholders in official method ymls that must be replaced with config roots.
CALIB_PATH_PLACEHOLDER = "calib data path"
EVAL_PATH_PLACEHOLDER = "eval data path"


def fill_template(
    template_path: Path,
    *,
    model_path: str,
    model_type: str,
    save_path: Path,
    calib_root: Path,
    eval_root: Path,
    base_seed: int,
    calib_auto_download: bool = False,
    eval_auto_download: bool = False,
    weight_bits: int | None = None,
) -> dict:
    """Fill in the official method yml template with common pipeline fields.

    Returns a yaml-ready dict; method-specific subclasses can mutate it
    further (e.g. add ``ignored_layers`` for AWQ) before it is written to disk.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(CALIB_PATH_PLACEHOLDER, str(calib_root))
    content = content.replace(EVAL_PATH_PLACEHOLDER, str(eval_root))
    data = yaml.safe_load(content)
    if data is None:
        data = {}
    if "model" not in data:
        data["model"] = {}
    data["model"]["path"] = model_path
    data["model"]["type"] = model_type
    if "save" not in data:
        data["save"] = {}
    data["save"]["save_path"] = str(save_path)
    data["save"]["save_fake"] = True
    if "base" not in data:
        data["base"] = {}
    data["base"]["seed"] = base_seed
    if "calib" in data:
        data["calib"]["download"] = calib_auto_download
    if "eval" in data:
        data["eval"]["download"] = eval_auto_download
    if weight_bits is not None and "quant" in data and "weight" in data["quant"]:
        data["quant"]["weight"]["bit"] = weight_bits
    return data


def dump_run_yml(run_yml: dict, out_path: Path) -> None:
    """Write the prepared run dict to disk for LightCompress to consume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(run_yml, f, allow_unicode=True, default_flow_style=False)
