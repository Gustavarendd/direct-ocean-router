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

    def is_safe(self, y: int, x: int, min_draft: float) -> bool:
        """Check if cell is safe for a vessel with given draft.
        
        GEBCO uses negative values for depth below sea level.
        A cell is safe if: depth <= -min_draft (water is deep enough)
        
        Args:
            y, x: Grid coordinates
            min_draft: Minimum required water depth in meters (positive value)
        """
        depth = float(self.depth[y, x])
        if depth == self.nodata:
            return False
        # GEBCO: negative = underwater, positive = above water
        # Safe if depth is negative (underwater) and abs(depth) >= draft
        return depth <= -min_draft

    def depth_penalty(self, y: int, x: int, min_draft: float, near_threshold_penalty: float = 0.0) -> float:
        """Calculate penalty for shallow water.
        
        Args:
            y, x: Grid coordinates
            min_draft: Minimum required water depth in meters (positive value)
            near_threshold_penalty: Penalty weight for near-threshold depths
        """
        depth = float(self.depth[y, x])
        if depth == self.nodata or depth > -min_draft:
            return float("inf")
        # Calculate slack (how much deeper than required)
        water_depth = -depth  # Convert to positive depth
        slack = max(water_depth - min_draft, 1e-3)
        return near_threshold_penalty / slack


def load_bathy(path: str | Path, nodata: Optional[int] = None) -> Bathy:
    if nodata is None:
        nodata = -32768
    return Bathy(Path(path), nodata=nodata)
