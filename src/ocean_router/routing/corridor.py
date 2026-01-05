"""Corridor builder that buffers macro waypoints into a raster mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from rasterio import features
import rasterio

from ocean_router.core.grid import GridSpec, window_from_bbox


@dataclass
class Corridor:
    mask: np.ndarray
    x_off: int
    y_off: int

    def contains(self, y: int, x: int) -> bool:
        return bool(self.mask[y - self.y_off, x - self.x_off])


def build_corridor(
    grid: GridSpec,
    start: Tuple[float, float],
    end: Tuple[float, float],
    land_mask: Optional[np.ndarray] = None,
    width_nm: float = 50.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Build a corridor mask between start and end points.
    
    Returns:
        corridor_mask: Binary mask where 1 = traversable
        x_off: X offset into global grid
        y_off: Y offset into global grid
    """
    # Create line between start and end
    line = LineString([start, end])
    
    # Buffer the line to create corridor (round caps to include endpoints)
    buffer_deg = width_nm / 60.0
    corridor_geom = line.buffer(buffer_deg, cap_style=1)  # round cap
    
    # Get bounding box
    minx, miny, maxx, maxy = corridor_geom.bounds
    
    # Add padding
    padding = buffer_deg * 0.5
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    
    # Clip to grid bounds
    minx = max(minx, grid.xmin)
    miny = max(miny, grid.ymin)
    maxx = min(maxx, grid.xmax)
    maxy = min(maxy, grid.ymax)
    
    # Convert to grid indices
    x_off = int((minx - grid.xmin) / grid.dx)
    y_off = int((grid.ymax - maxy) / grid.dy)
    x_end = int((maxx - grid.xmin) / grid.dx) + 1
    y_end = int((grid.ymax - miny) / grid.dy) + 1
    
    # Clip to valid range
    x_off = max(0, x_off)
    y_off = max(0, y_off)
    x_end = min(grid.width, x_end)
    y_end = min(grid.height, y_end)
    
    w = x_end - x_off
    h = y_end - y_off
    
    # Create transform for this window
    window_minx = grid.xmin + x_off * grid.dx
    window_maxy = grid.ymax - y_off * grid.dy
    transform = rasterio.transform.from_origin(window_minx, window_maxy, grid.dx, grid.dy)
    
    # Rasterize corridor geometry
    mask = features.rasterize(
        [(corridor_geom, 1)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    
    # Remove land from corridor if land mask provided
    if land_mask is not None:
        land_window = land_mask[y_off:y_end, x_off:x_end]
        mask[land_window > 0] = 0
    
    return mask, x_off, y_off


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

        for iy, lat in enumerate(ys):
            for ix, lon in enumerate(xs):
                if corridor_geom.contains(Point(lon, lat)):
                    mask[iy, ix] = 1
        return Corridor(mask=mask, x_off=x_off, y_off=y_off)

    def _nm_to_deg(self, nm: float) -> float:
        return nm / 60.0
