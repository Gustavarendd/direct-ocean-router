"""Land mask helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
import numba

from ocean_router.core.memmaps import MemMapLoader
from ocean_router.core.tiled_raster import TiledRaster, TiledRasterArray


@numba.jit
def _proximity_penalty_jit(dist: float, penalty_weight: float, max_distance_cells: int) -> float:
    """JIT-compiled helper for land proximity penalty calculation."""
    if dist >= max_distance_cells:
        return 0.0
    # Penalty decreases linearly with distance
    # At dist=0 (land edge): full penalty
    # At dist=max_distance_cells: no penalty
    return penalty_weight * (1.0 - dist / max_distance_cells)


class LandMask:
    def __init__(
        self,
        path: Optional[str | Path],
        buffered_path: Optional[str | Path] = None,
        distance_cache_path: Optional[str | Path] = None,
        tiles_meta_path: Optional[Path] = None,
        buffered_tiles_meta_path: Optional[Path] = None,
        distance_tiles_meta_path: Optional[Path] = None,
    ):
        self.path = Path(path) if path else None
        self.buffered_path = Path(buffered_path) if buffered_path else None
        self._tiled = TiledRaster.open(tiles_meta_path) if tiles_meta_path else None
        self._tiled_buffered = (
            TiledRaster.open(buffered_tiles_meta_path) if buffered_tiles_meta_path else None
        )
        self._distance_tiles = (
            TiledRaster.open(distance_tiles_meta_path) if distance_tiles_meta_path else None
        )
        self._base = MemMapLoader(self.path, mode="r", dtype=np.uint8) if self.path else None
        self._buffered = MemMapLoader(self.buffered_path, mode="r", dtype=np.uint8) if self.buffered_path else None
        self._distance_from_land = None
        if self._tiled is not None:
            self._base_array = TiledRasterArray(self._tiled)
        else:
            self._base_array = self._base.array if self._base else None
        if self._tiled_buffered is not None:
            self._buffered_array = TiledRasterArray(self._tiled_buffered)
        else:
            self._buffered_array = self._buffered.array if self._buffered else None
        
        # Distance cache path - derive from base path if not provided
        if distance_cache_path:
            self._distance_cache_path = Path(distance_cache_path)
        elif self.path:
            self._distance_cache_path = self.path.parent / self.path.name.replace('.npy', '_distance.npy')
        else:
            self._distance_cache_path = None

    @property
    def base(self) -> np.memmap:
        return self._base_array

    @property
    def buffered(self) -> np.memmap:
        if self._buffered_array is None:
            return self.base
        return self._buffered_array

    @property
    def shape(self) -> tuple[int, int]:
        if self._tiled is not None:
            return self._tiled.shape
        return self.base.shape

    def sample(self, y: int, x: int, buffered: bool = False) -> int:
        raster = self._tiled_buffered if buffered and self._tiled_buffered is not None else self._tiled
        if raster is not None:
            return int(raster.sample_xy(x, y))
        array = self.buffered if buffered else self.base
        return int(array[y, x])

    def window(self, y_off: int, x_off: int, height: int, width: int, buffered: bool = False) -> np.ndarray:
        raster = self._tiled_buffered if buffered and self._tiled_buffered is not None else self._tiled
        if raster is not None:
            return raster.window(y_off, x_off, height, width)
        array = self.buffered if buffered else self.base
        return array[y_off:y_off + height, x_off:x_off + width]

    @property
    def distance_from_land(self) -> np.ndarray:
        """Get or load distance transform from land (in grid cells).
        
        Loads from disk cache if available, otherwise computes and caches.
        """
        if self._distance_from_land is None:
            if self._distance_tiles is not None:
                self._distance_from_land = TiledRasterArray(self._distance_tiles)
                return self._distance_from_land
            # Try to load from cache
            if self._distance_cache_path and self._distance_cache_path.exists():
                print(f"[LandMask] Loading distance transform from cache: {self._distance_cache_path}")
                self._distance_from_land = np.load(self._distance_cache_path, mmap_mode='r')
                print(f"[LandMask] Distance transform loaded. Shape: {self._distance_from_land.shape}")
            else:
                # Compute and save to cache
                print("[LandMask] Computing distance transform from land (this is a one-time operation)...")
                if self._tiled is not None:
                    land_bool = self._tiled.window(0, 0, self.shape[0], self.shape[1]).astype(bool)
                else:
                    land_bool = self.base.astype(bool)
                # distance_transform_edt computes distance from False (ocean) to True (land)
                # We want distance from land, so invert
                dist = distance_transform_edt(~land_bool).astype(np.float32)
                print(f"[LandMask] Distance transform computed. Shape: {dist.shape}, Max distance: {dist.max():.1f} cells")
                
                # Save to cache
                if self._distance_cache_path is None:
                    raise ValueError("distance_cache_path is required when computing distance transform")
                print(f"[LandMask] Saving distance transform to cache: {self._distance_cache_path}")
                np.save(self._distance_cache_path, dist)
                
                # Also save metadata
                meta_path = self._distance_cache_path.with_suffix('.meta.json')
                meta = {
                    'shape': list(dist.shape),
                    'dtype': str(dist.dtype),
                    'max_distance': float(dist.max())
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                # Load as memory-mapped for efficiency
                self._distance_from_land = np.load(self._distance_cache_path, mmap_mode='r')
                print("[LandMask] Distance transform cached and ready.")
                
        return self._distance_from_land

    def distance_window(self, y_off: int, x_off: int, height: int, width: int) -> np.ndarray:
        dist = self.distance_from_land
        if isinstance(dist, TiledRasterArray):
            return dist[y_off:y_off + height, x_off:x_off + width]
        return dist[y_off:y_off + height, x_off:x_off + width]

    def proximity_penalty(self, y: int, x: int, penalty_weight: float = 1.0, max_distance_cells: int = 50) -> float:
        """Calculate penalty based on proximity to land.
        
        Args:
            y, x: Grid coordinates
            penalty_weight: Weight multiplier for the penalty
            max_distance_cells: Distance beyond which penalty is 0
            
        Returns:
            Penalty that decreases with distance from land
        """
        dist = self.distance_from_land[y, x]
        return _proximity_penalty_jit(dist, penalty_weight, max_distance_cells)

    def dilate(self, iterations: int = 1) -> np.ndarray:
        mask = self.base.astype(bool)
        dilated = binary_dilation(mask, iterations=iterations)
        return dilated.astype(np.uint8)

    def save_buffered(self, path: str | Path, iterations: int = 1) -> None:
        dilated = self.dilate(iterations)
        arr = np.memmap(path, mode="w+", dtype=np.uint8, shape=dilated.shape)
        arr[:] = dilated
        arr.flush()
