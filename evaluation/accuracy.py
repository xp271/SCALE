"""Run compute_accuracy.py on a syco output pkl and parse its console output."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TIMEOUT_COMPUTE_ACCURACY = 60


def run_compute_accuracy(pkl_path: str, script_dir: Path) -> dict | None:
    """Run compute_accuracy.py on pkl; return parsed metrics if opinion_only."""
    cmd = [sys.executable, "compute_accuracy.py", pkl_path]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_COMPUTE_ACCURACY,
        )
        out = (result.stdout or "") + (result.stderr or "")
        metrics: dict = {}
        # opinion_only: 正确 / 顺从 / 其他
        m = re.search(r"正确 \(选对\):\s+([\d.]+)", out)
        if m:
            metrics["correct_pct"] = float(m.group(1))
        m = re.search(r"顺从错误观点:\s+([\d.]+)", out)
        if m:
            metrics["sycophantic_pct"] = float(m.group(1))
        m = re.search(r"其他:\s+([\d.]+)", out)
        if m:
            metrics["other_pct"] = float(m.group(1))
        if not metrics:
            m = re.search(r"准确率:\s+([\d.]+)", out)
            if m:
                metrics["accuracy"] = float(m.group(1))
        return metrics if metrics else None
    except Exception as e:
        print(f"compute_accuracy failed: {e}", file=sys.stderr)
        return None
