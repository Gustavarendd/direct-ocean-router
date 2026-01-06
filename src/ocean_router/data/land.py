"""Land mask helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ocean_router.core.memmaps import MemMapLoader


class LandMask:
    def __init__(self, path: str | Path, buffered_path: Optional[str | Path] = None, 
                 distance_cache_path: Optional[str | Path] = None):
        self.path = Path(path)
        self.buffered_path = Path(buffered_path) if buffered_path else None
        self._base = MemMapLoader(self.path, mode="r", dtype=np.uint8)
        self._buffered = MemMapLoader(self.buffered_path, mode="r", dtype=np.uint8) if self.buffered_path else None
        self._distance_from_land = None
        
        # Distance cache path - derive from base path if not provided
        if distance_cache_path:
            self._distance_cache_path = Path(distance_cache_path)
        else:
            self._distance_cache_path = self.path.parent / self.path.name.replace('.npy', '_distance.npy')

    @property
    def base(self) -> np.memmap:
        return self._base.array

    @property
    def buffered(self) -> np.memmap:
        if self._buffered is None:
            return self.base
        return self._buffered.array

    @property
    def distance_from_land(self) -> np.ndarray:
        """Get or load distance transform from land (in grid cells).
        
        Loads from disk cache if available, otherwise computes and caches.
        """
        if self._distance_from_land is None:
            # Try to load from cache
            if self._distance_cache_path.exists():
                print(f"[LandMask] Loading distance transform from cache: {self._distance_cache_path}")
                self._distance_from_land = np.load(self._distance_cache_path, mmap_mode='r')
                print(f"[LandMask] Distance transform loaded. Shape: {self._distance_from_land.shape}")
            else:
                # Compute and save to cache
                print("[LandMask] Computing distance transform from land (this is a one-time operation)...")
                land_bool = self.base.astype(bool)
                # distance_transform_edt computes distance from False (ocean) to True (land)
                # We want distance from land, so invert
                dist = distance_transform_edt(~land_bool).astype(np.float32)
                print(f"[LandMask] Distance transform computed. Shape: {dist.shape}, Max distance: {dist.max():.1f} cells")
                
                # Save to cache
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
        if dist >= max_distance_cells:
            return 0.0
        # Penalty decreases linearly with distance
        # At dist=0 (land edge): full penalty
        # At dist=max_distance_cells: no penalty
        return penalty_weight * (1.0 - dist / max_distance_cells)

    def dilate(self, iterations: int = 1) -> np.ndarray:
        mask = self.base.astype(bool)
        dilated = binary_dilation(mask, iterations=iterations)
        return dilated.astype(np.uint8)

    def save_buffered(self, path: str | Path, iterations: int = 1) -> None:
        dilated = self.dilate(iterations)
        arr = np.memmap(path, mode="w+", dtype=np.uint8, shape=dilated.shape)
        arr[:] = dilated
        arr.flush()
