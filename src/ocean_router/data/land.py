"""Land mask helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation

from ocean_router.core.memmaps import MemMapLoader


class LandMask:
    def __init__(self, path: str | Path, buffered_path: Optional[str | Path] = None):
        self.path = Path(path)
        self.buffered_path = Path(buffered_path) if buffered_path else None
        self._base = MemMapLoader(self.path, mode="r", dtype=np.uint8)
        self._buffered = MemMapLoader(self.buffered_path, mode="r", dtype=np.uint8) if self.buffered_path else None

    @property
    def base(self) -> np.memmap:
        return self._base.array

    @property
    def buffered(self) -> np.memmap:
        if self._buffered is None:
            return self.base
        return self._buffered.array

    def dilate(self, iterations: int = 1) -> np.ndarray:
        mask = self.base.astype(bool)
        dilated = binary_dilation(mask, iterations=iterations)
        return dilated.astype(np.uint8)

    def save_buffered(self, path: str | Path, iterations: int = 1) -> None:
        dilated = self.dilate(iterations)
        arr = np.memmap(path, mode="w+", dtype=np.uint8, shape=dilated.shape)
        arr[:] = dilated
        arr.flush()
