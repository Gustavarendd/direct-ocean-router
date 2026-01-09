"""Bathymetry lookups and validations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numba

from ocean_router.core.memmaps import MemMapLoader
from ocean_router.core.tiled_raster import TiledRaster, TiledRasterArray


@numba.jit
def _depth_penalty_jit(depth: float, nodata: int, y: int, x: int, min_draft: float, near_threshold_penalty: float = 0.0) -> float:
    """JIT-compiled helper for depth penalty calculation."""
    if depth == nodata or depth > -min_draft:
        return float("inf")
    # Calculate slack (how much deeper than required)
    water_depth = -depth  # Convert to positive depth
    slack = max(water_depth - min_draft, 1e-3)
    return near_threshold_penalty / slack


@dataclass
class Bathy:
    path: Optional[Path]
    nodata: int | float
    tiles_meta_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self._tiled = None
        self.loader = None
        if self.tiles_meta_path is not None:
            self._tiled = TiledRaster.open(self.tiles_meta_path)
            self._depth = TiledRasterArray(self._tiled)
        elif self.path is not None:
            self.loader = MemMapLoader(self.path, dtype=np.int16)
            self._depth = self.loader.array
        else:
            raise ValueError("Bathy requires a path or tiles_meta_path")

    @property
    def depth(self) -> np.memmap:
        return self._depth

    def depth_window(self, y_off: int, x_off: int, height: int, width: int) -> np.ndarray:
        if self._tiled is not None:
            return self._tiled.window(y_off, x_off, height, width)
        return self._depth[y_off:y_off + height, x_off:x_off + width]

    def depth_value(self, y: int, x: int) -> float:
        if self._tiled is not None:
            return float(self._tiled.sample_xy(x, y))
        return float(self._depth[y, x])

    def is_safe(self, y: int, x: int, min_draft: float) -> bool:
        """Check if cell is safe for a vessel with given draft.
        
        GEBCO uses negative values for depth below sea level.
        A cell is safe if: depth <= -min_draft (water is deep enough)
        
        Args:
            y, x: Grid coordinates
            min_draft: Minimum required water depth in meters (positive value)
        """
        depth = self.depth_value(y, x)
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
        depth = self.depth_value(y, x)
        return _depth_penalty_jit(depth, self.nodata, y, x, min_draft, near_threshold_penalty)


def load_bathy(path: str | Path, nodata: Optional[int] = None) -> Bathy:
    if nodata is None:
        nodata = -32768
    return Bathy(Path(path), nodata=nodata)


def load_bathy_tiles(meta_path: Path, nodata: Optional[int] = None) -> Bathy:
    if nodata is None:
        nodata = -32768
    return Bathy(path=None, nodata=nodata, tiles_meta_path=meta_path)
