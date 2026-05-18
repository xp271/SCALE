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
        # opinion_only: correct / sycophantic / other
        m = re.search(r"Correct:\s+([\d.]+)", out)
        if m:
            metrics["correct_pct"] = float(m.group(1))
        m = re.search(r"Sycophantic \(wrong\):\s+([\d.]+)", out)
        if m:
            metrics["sycophantic_pct"] = float(m.group(1))
        m = re.search(r"Other:\s+([\d.]+)", out)
        if m:
            metrics["other_pct"] = float(m.group(1))
        if not metrics:
            m = re.search(r"Accuracy:\s+([\d.]+)", out)
            if m:
                metrics["accuracy"] = float(m.group(1))
        return metrics if metrics else None
    except Exception as e:
        print(f"compute_accuracy failed: {e}", file=sys.stderr)
        return None
