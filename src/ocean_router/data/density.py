"""Ship density heatmap lookups."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ocean_router.core.memmaps import MemMapLoader


class Density:
    def __init__(self, path: str | Path):
        self.loader = MemMapLoader(path, dtype=np.float32)

    @property
    def array(self) -> np.memmap:
        return self.loader.array

    def bias(self, y: int, x: int, weight: float) -> float:
        return -weight * float(self.array[y, x])
