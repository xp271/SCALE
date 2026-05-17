"""Reserved for future CLI helpers.

CLI parsing for ``run_pipeline.py`` moved to :mod:`config.parser`. This module
is kept as an explicit placeholder so other packages can extend it without
collisions; if you need argparse helpers shared across multiple entries,
add them here.
"""
from __future__ import annotations

__all__: list[str] = []
