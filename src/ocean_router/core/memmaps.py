"""Helpers for reading and writing memmapped arrays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


class MemMapLoader:
    """Lazy loader for memmapped grids."""

    def __init__(self, path: str | Path, mode: str = "r", dtype: Any | None = None, shape: tuple[int, ...] | None = None):
        self.path = Path(path)
        self.mode = mode
        self.dtype = dtype
        self.shape = shape
        self._arr: np.memmap | None = None
        
        # Try to load shape from metadata file if not provided
        if self.shape is None:
            meta_path = self.path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.shape = tuple(meta["shape"])
                    if self.dtype is None:
                        self.dtype = np.dtype(meta["dtype"])

    @property
    def array(self) -> np.memmap:
        if self._arr is None:
            kwargs = {}
            if self.dtype is not None:
                kwargs["dtype"] = self.dtype
            if self.shape is not None:
                kwargs["shape"] = self.shape
            self._arr = np.memmap(self.path, mode=self.mode, **kwargs)
        return self._arr

    def __array__(self) -> np.ndarray:  # pragma: no cover - passthrough
        return np.asarray(self.array)


def save_memmap(path: str | Path, array: np.ndarray, dtype: Any | None = None) -> None:
    """Save array as memmap with metadata for shape/dtype."""
    path = Path(path)
    if dtype:
        array = array.astype(dtype)
    
    # Save the array data
    arr = np.memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
    arr[:] = array
    arr.flush()
    
    # Save metadata (shape and dtype) for later loading
    meta_path = path.with_suffix(".meta.json")
    meta = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
