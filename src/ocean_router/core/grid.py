"""Grid utilities for converting between geographic coordinates and array indices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json

import numpy as np


@dataclass(slots=True)
class GridSpec:
    """Definition of the global routing grid.

    Attributes:
        crs: Coordinate reference system string (usually EPSG:4326).
        dx: Cell width in degrees longitude.
        dy: Cell height in degrees latitude.
        xmin: Minimum longitude of the grid extent.
        ymax: Maximum latitude of the grid extent (top edge).
        width: Number of columns in the grid.
        height: Number of rows in the grid.
    """

    crs: str
    dx: float
    dy: float
    xmin: float
    ymax: float
    width: int
    height: int

    @classmethod
    def from_file(cls, path: str | Path) -> "GridSpec":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @property
    def xmax(self) -> float:
        return self.xmin + self.dx * self.width

    @property
    def ymin(self) -> float:
        return self.ymax - self.dy * self.height

    @property
    def lon_span(self) -> float:
        return self.xmax - self.xmin

    def wrap_lon(self, lon: float) -> float:
        span = self.lon_span
        if span <= 0:
            return lon
        return ((lon - self.xmin) % span) + self.xmin

    def wrap_x(self, x: int) -> int:
        if self.width <= 0:
            return x
        return x % self.width

    def lonlat_to_xy(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert lon/lat to integer grid indices (x, y).

        Y is counted from the north (row-major), so higher latitudes have smaller y.
        """
        lon = self.wrap_lon(lon)
        x = int(np.floor((lon - self.xmin) / self.dx))
        y = int(np.floor((self.ymax - lat) / self.dy))
        return x, y

    def xy_to_lonlat(self, x: int, y: int) -> Tuple[float, float]:
        x = self.wrap_x(x)
        lon = self.xmin + (x + 0.5) * self.dx
        lat = self.ymax - (y + 0.5) * self.dy
        return lon, lat

    def clip_indices(self, x: int, y: int) -> Tuple[int, int]:
        cx = min(max(x, 0), self.width - 1)
        cy = min(max(y, 0), self.height - 1)
        return cx, cy

    def valid_index(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


def window_from_bbox(grid: GridSpec, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> tuple[int, int, int, int]:
    """Compute the grid window covering a geographic bounding box.

    Returns (x_off, y_off, width, height) suitable for windowed array extraction.
    """
    x0, y1 = grid.lonlat_to_xy(min_lon, max_lat)
    x1, y0 = grid.lonlat_to_xy(max_lon, min_lat)
    x_off = max(0, min(x0, x1))
    y_off = max(0, min(y0, y1))
    w = min(grid.width, max(x0, x1) + 1) - x_off
    h = min(grid.height, max(y0, y1) + 1) - y_off
    return x_off, y_off, w, h
