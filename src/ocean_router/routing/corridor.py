"""Corridor builder that buffers macro waypoints into a raster mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from ocean_router.core.grid import GridSpec, window_from_bbox


@dataclass
class Corridor:
    mask: np.ndarray
    x_off: int
    y_off: int

    def contains(self, y: int, x: int) -> bool:
        return bool(self.mask[y - self.y_off, x - self.x_off])


@dataclass
class CorridorBuilder:
    grid: GridSpec
    offshore_buffer_nm: float
    chokepoint_buffer_nm: float

    def build(self, waypoints: Iterable[Tuple[float, float]], min_width_nm: float = 25.0) -> Corridor:
        line = LineString([(lon, lat) for lon, lat in waypoints])
        if line.is_empty:
            raise ValueError("Waypoints cannot be empty")
        buffer_deg = self._nm_to_deg(self.offshore_buffer_nm)
        narrow_deg = self._nm_to_deg(self.chokepoint_buffer_nm)
        corridor_geom = unary_union([
            line.buffer(buffer_deg, cap_style=1),
            line.buffer(narrow_deg, cap_style=2),
        ])
        min_buffer_deg = self._nm_to_deg(min_width_nm)
        corridor_geom = corridor_geom.buffer(0).buffer(min_buffer_deg)
        minx, miny, maxx, maxy = corridor_geom.bounds
        x_off, y_off, w, h = window_from_bbox(self.grid, minx, miny, maxx, maxy)
        mask = np.zeros((h, w), dtype=np.uint8)
        xs = np.linspace(minx, maxx, w, endpoint=False) + self.grid.dx / 2
        ys = np.linspace(maxy, miny, h, endpoint=False) + (-self.grid.dy / 2)
        from shapely.geometry import Point

        for iy, lat in enumerate(ys):
            for ix, lon in enumerate(xs):
                if corridor_geom.contains(Point(lon, lat)):
                    mask[iy, ix] = 1
        return Corridor(mask=mask, x_off=x_off, y_off=y_off)

    def _nm_to_deg(self, nm: float) -> float:
        return nm / 60.0
