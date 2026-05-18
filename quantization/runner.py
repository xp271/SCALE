"""Invoke LightCompress via torchrun for one (model, method, bits) task."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_lightcompress(
    llmc_root: Path,
    run_yml_path: Path,
    task_id: str,
    nproc_per_node: int,
    cuda_devices: str | None = None,
) -> bool:
    """Launch LightCompress on the prepared run yml. Returns True on success."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(llmc_root)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if cuda_devices is not None:
        # Visible GPUs for this LightCompress step only
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    port = 29500 + (hash(task_id) % 30000)
    if port < 10000:
        port += 20000
    endpoint = f"127.0.0.1:{port}"
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", "1",
        "--nproc_per_node", str(nproc_per_node),
        "--rdzv_id", task_id,
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", endpoint,
        str(llmc_root / "llmc" / "__main__.py"),
        "--config", str(run_yml_path),
        "--task_id", task_id,
    ]
    try:
        ret = subprocess.run(cmd, env=env, cwd=str(llmc_root))
        return ret.returncode == 0
    except Exception as e:
        print(f"LightCompress run failed: {e}", file=sys.stderr)
        return False
