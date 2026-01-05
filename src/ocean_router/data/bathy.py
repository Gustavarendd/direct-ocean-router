"""Bathymetry lookups and validations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ocean_router.core.memmaps import MemMapLoader


@dataclass
class Bathy:
    path: Path
    nodata: int | float

    def __post_init__(self) -> None:
        self.loader = MemMapLoader(self.path, dtype=np.int16)

    @property
    def depth(self) -> np.memmap:
        return self.loader.array

    def is_safe(self, y: int, x: int, min_depth: float) -> bool:
        depth = float(self.depth[y, x])
        if depth == self.nodata:
            return False
        return depth >= min_depth

    def depth_penalty(self, y: int, x: int, min_depth: float, near_threshold_penalty: float = 0.0) -> float:
        depth = float(self.depth[y, x])
        if depth == self.nodata or depth < min_depth:
            return float("inf")
        slack = max(depth - min_depth, 1e-3)
        return near_threshold_penalty / slack


def load_bathy(path: str | Path, nodata: Optional[int] = None) -> Bathy:
    if nodata is None:
        nodata = -32768
    return Bathy(Path(path), nodata=nodata)
