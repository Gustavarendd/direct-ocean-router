"""Dependency wiring for API service."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ocean_router.core.grid import GridSpec


@lru_cache(maxsize=1)
def get_grid_spec() -> GridSpec:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "grid_1nm.json"
    return GridSpec.from_file(cfg_path)
