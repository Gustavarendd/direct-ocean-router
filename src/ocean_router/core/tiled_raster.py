"""Tile-backed raster loading with LRU caching."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np

from ocean_router.core.geodesy import rhumb_distance_nm, rhumb_interpolate


@dataclass(frozen=True)
class RasterTileMeta:
    xmin: float
    ymax: float
    dx: float
    dy: float
    width: int
    height: int
    tile_size: int
    dtype: str
    nodata: float | int
    tiles_x: int
    tiles_y: int
    tiles_dir: Path


class TiledRaster:
    """Tile-backed raster reader that loads tiles on demand."""

    def __init__(self, meta: RasterTileMeta, max_cache_tiles: int = 64) -> None:
        self.meta = meta
        self._cache: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_tiles = max(1, max_cache_tiles)

    @classmethod
    def open(cls, meta_path: Path, max_cache_tiles: int = 64) -> "TiledRaster":
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tiles_dir = Path(payload["tiles_dir"])
        if not tiles_dir.is_absolute():
            tiles_dir = meta_path.parent / tiles_dir
        meta = RasterTileMeta(
            xmin=float(payload["xmin"]),
            ymax=float(payload["ymax"]),
            dx=float(payload["dx"]),
            dy=float(payload["dy"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            tile_size=int(payload["tile_size"]),
            dtype=str(payload["dtype"]),
            nodata=payload.get("nodata", 0),
            tiles_x=int(payload["tiles_x"]),
            tiles_y=int(payload["tiles_y"]),
            tiles_dir=tiles_dir,
        )
        return cls(meta=meta, max_cache_tiles=max_cache_tiles)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.meta.height, self.meta.width)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.meta.dtype)

    def lonlat_to_xy(self, lon: float, lat: float) -> tuple[int, int]:
        x = int(np.floor((lon - self.meta.xmin) / self.meta.dx))
        y = int(np.floor((self.meta.ymax - lat) / self.meta.dy))
        return x, y

    def _tile_path(self, tx: int, ty: int) -> Path:
        return self.meta.tiles_dir / f"tile_{tx}_{ty}.npy"

    def _load_tile(self, tx: int, ty: int) -> np.ndarray:
        key = (tx, ty)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        tile_path = self._tile_path(tx, ty)
        arr = np.load(tile_path, mmap_mode="r")
        self._cache[key] = arr
        if len(self._cache) > self._max_cache_tiles:
            self._cache.popitem(last=False)
        return arr

    def cache_info(self) -> dict[str, Any]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "max_size": self._max_cache_tiles,
        }

    def sample_xy(self, x: int, y: int) -> float | int:
        if x < 0 or y < 0 or x >= self.meta.width or y >= self.meta.height:
            return self.meta.nodata
        tx = x // self.meta.tile_size
        ty = y // self.meta.tile_size
        tile = self._load_tile(tx, ty)
        local_x = x - tx * self.meta.tile_size
        local_y = y - ty * self.meta.tile_size
        if local_y >= tile.shape[0] or local_x >= tile.shape[1]:
            return self.meta.nodata
        return tile[local_y, local_x]

    def sample(self, lon: float, lat: float) -> float | int:
        x, y = self.lonlat_to_xy(lon, lat)
        return self.sample_xy(x, y)

    def window(self, y_off: int, x_off: int, height: int, width: int) -> np.ndarray:
        out = np.full((height, width), self.meta.nodata, dtype=self.dtype)
        if height <= 0 or width <= 0:
            return out
        y0 = max(0, y_off)
        x0 = max(0, x_off)
        y1 = min(self.meta.height, y_off + height)
        x1 = min(self.meta.width, x_off + width)
        if y1 <= y0 or x1 <= x0:
            return out
        tile_size = self.meta.tile_size
        tx0 = x0 // tile_size
        tx1 = (x1 - 1) // tile_size
        ty0 = y0 // tile_size
        ty1 = (y1 - 1) // tile_size
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = self._load_tile(tx, ty)
                tile_y0 = ty * tile_size
                tile_x0 = tx * tile_size
                overlap_y0 = max(y0, tile_y0)
                overlap_x0 = max(x0, tile_x0)
                overlap_y1 = min(y1, tile_y0 + tile.shape[0])
                overlap_x1 = min(x1, tile_x0 + tile.shape[1])
                if overlap_y1 <= overlap_y0 or overlap_x1 <= overlap_x0:
                    continue
                out_y0 = overlap_y0 - y_off
                out_x0 = overlap_x0 - x_off
                out_y1 = out_y0 + (overlap_y1 - overlap_y0)
                out_x1 = out_x0 + (overlap_x1 - overlap_x0)
                tile_y_start = overlap_y0 - tile_y0
                tile_x_start = overlap_x0 - tile_x0
                out[out_y0:out_y1, out_x0:out_x1] = tile[
                    tile_y_start : tile_y_start + (overlap_y1 - overlap_y0),
                    tile_x_start : tile_x_start + (overlap_x1 - overlap_x0),
                ]
        return out

    def sample_segment(
        self,
        lon0: float,
        lat0: float,
        lon1: float,
        lat1: float,
        step_nm: Optional[float] = None,
        step_m: Optional[float] = None,
    ) -> list[float | int]:
        if step_nm is None:
            if step_m is None:
                step_nm = 0.25
            else:
                step_nm = step_m / 1852.0
        total_nm = rhumb_distance_nm(lon0, lat0, lon1, lat1)
        if total_nm <= 0:
            return [self.sample(lon0, lat0)]
        steps = max(1, int(total_nm / max(step_nm, 1e-6)))
        samples: list[float | int] = []
        for i in range(steps + 1):
            frac = i / steps
            lon, lat = rhumb_interpolate(lon0, lat0, lon1, lat1, frac)
            samples.append(self.sample(lon, lat))
        return samples


class TiledRasterArray:
    """Array-like view backed by a TiledRaster."""

    def __init__(self, raster: TiledRaster) -> None:
        self._raster = raster

    @property
    def shape(self) -> tuple[int, int]:
        return self._raster.shape

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            y_key, x_key = key
            if isinstance(y_key, int) and isinstance(x_key, slice):
                x0 = x_key.start or 0
                x1 = x_key.stop if x_key.stop is not None else self.shape[1]
                return self._raster.window(y_key, x0, 1, x1 - x0)[0]
            if isinstance(y_key, slice) and isinstance(x_key, int):
                y0 = y_key.start or 0
                y1 = y_key.stop if y_key.stop is not None else self.shape[0]
                return self._raster.window(y0, x_key, y1 - y0, 1)[:, 0]
            if isinstance(y_key, slice) and isinstance(x_key, slice):
                y0 = y_key.start or 0
                y1 = y_key.stop if y_key.stop is not None else self.shape[0]
                x0 = x_key.start or 0
                x1 = x_key.stop if x_key.stop is not None else self.shape[1]
                return self._raster.window(y0, x0, y1 - y0, x1 - x0)
            return self._raster.sample_xy(int(x_key), int(y_key))
        if isinstance(key, slice):
            y0 = key.start or 0
            y1 = key.stop if key.stop is not None else self.shape[0]
            return self._raster.window(y0, 0, y1 - y0, self.shape[1])
        if isinstance(key, int):
            return self._raster.window(key, 0, 1, self.shape[1])[0]
        raise TypeError(f"Unsupported index type: {type(key)!r}")
