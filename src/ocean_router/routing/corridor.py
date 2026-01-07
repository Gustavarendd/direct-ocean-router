"""Corridor builder that buffers macro waypoints into a raster mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, Sequence, TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from rasterio import features
import rasterio
from scipy import ndimage

from ocean_router.core.grid import GridSpec, window_from_bbox
from ocean_router.data.canals import canal_mask_window

if TYPE_CHECKING:
    from ocean_router.data.canals import Canal


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
    canals: Optional[Sequence["Canal"]] = None,
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

    # Carve canals back into the mask if provided
    if canals:
        canal_mask = canal_mask_window(canals, grid, x_off, y_off, h, w)
        if canal_mask.size:
            mask[canal_mask > 0] = 1
    
    return mask, x_off, y_off


def build_corridor_with_arrays(
    grid: GridSpec,
    start: Tuple[float, float],
    end: Tuple[float, float],
    land_mask: Optional[np.ndarray] = None,
    width_nm: float = 50.0,
    context: Optional[object] = None,
    min_draft: float = 10.0,
    weights: Optional[object] = None,
    canals: Optional[Sequence["Canal"]] = None,
) -> Tuple[np.ndarray, int, int, Dict[str, Any]]:
    """Build corridor and precompute per-corridor arrays to speed up A*.

    Returns: (mask, x_off, y_off, precomputed_dict)
    precomputed_dict may contain:
      - blocked_mask: bool array
      - depth_penalty: float array
      - land_prox_penalty: float array
      - tss_in_or_near: bool array
      - tss_direction: float array (or -1)
    """
    mask, x_off, y_off = build_corridor(
        grid,
        start,
        end,
        land_mask=land_mask,
        width_nm=width_nm,
        canals=canals,
    )
    canal_mask = None
    if canals:
        canal_mask = canal_mask_window(canals, grid, x_off, y_off, mask.shape[0], mask.shape[1])

    pre = precompute_corridor_arrays(
        mask,
        x_off,
        y_off,
        context=context,
        min_draft=min_draft,
        weights=weights,
        override_mask=canal_mask,
    )

    return mask, x_off, y_off, pre


def precompute_corridor_arrays(
    mask: np.ndarray,
    x_off: int,
    y_off: int,
    context: Optional[object] = None,
    min_draft: float = 10.0,
    weights: Optional[object] = None,
    tss_radius: int = 2,
    override_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Vectorized precompute of per-corridor arrays for faster A*."""
    pre: Dict[str, Any] = {}
    if context is None:
        return pre

    h, w = mask.shape
    if h == 0 or w == 0:
        pre["blocked_mask"] = np.zeros((h, w), dtype=bool)
        pre["depth_penalty"] = np.zeros((h, w), dtype=float)
        pre["land_prox_penalty"] = np.zeros((h, w), dtype=float)
        pre["tss_in_or_near"] = np.zeros((h, w), dtype=bool)
        pre["tss_direction"] = np.full((h, w), -1.0, dtype=float)
        return pre

    mask_bool = mask.astype(bool)
    y1 = y_off + h
    x1 = x_off + w

    depth_window = context.bathy.depth[y_off:y1, x_off:x1]
    nodata = context.bathy.nodata
    blocked_depth = (depth_window == nodata) | (depth_window > -min_draft)
    if override_mask is not None and override_mask.size:
        blocked_depth = blocked_depth & ~override_mask.astype(bool)
    blocked_mask = blocked_depth.copy()

    if context.tss is not None and context.tss.sepzone_mask is not None:
        sep_window = context.tss.sepzone_mask[y_off:y1, x_off:x1] > 0
        blocked_mask |= sep_window

    blocked_mask &= mask_bool
    if override_mask is not None and override_mask.size:
        blocked_mask[override_mask.astype(bool)] = False

    depth_penalty = np.zeros((h, w), dtype=float)
    if weights is not None:
        safe = ~blocked_depth
        if safe.any():
            water_depth = -depth_window.astype(np.float32)
            slack = np.maximum(water_depth - min_draft, 1e-3)
            depth_penalty = weights.near_shore_depth_penalty / slack
            depth_penalty[~safe] = 0.0
        depth_penalty[~mask_bool] = 0.0
        if override_mask is not None and override_mask.size:
            depth_penalty[override_mask.astype(bool)] = 0.0

    land_prox_penalty = np.zeros((h, w), dtype=float)
    if context.land is not None and weights is not None:
        max_cells = int(weights.land_proximity_max_distance_cells)
        if max_cells > 0 and weights.land_proximity_penalty != 0:
            dist = context.land.distance_from_land[y_off:y1, x_off:x1].astype(np.float32)
            land_prox_penalty = weights.land_proximity_penalty * (1.0 - dist / max_cells)
            land_prox_penalty[dist >= max_cells] = 0.0
        land_prox_penalty[~mask_bool] = 0.0
        if override_mask is not None and override_mask.size:
            land_prox_penalty[override_mask.astype(bool)] = 0.0

    tss_in_or_near = np.zeros((h, w), dtype=bool)
    tss_direction = np.full((h, w), -1.0, dtype=float)
    if context.tss is not None:
        tss_direction = context.tss.direction_field[y_off:y1, x_off:x1].astype(np.float32)
        if tss_radius > 0:
            pad = tss_radius
            lane_mask = context.tss.lane_mask
            wy0 = max(0, y_off - pad)
            wy1 = min(lane_mask.shape[0], y_off + h + pad)
            wx0 = max(0, x_off - pad)
            wx1 = min(lane_mask.shape[1], x_off + w + pad)
            lane_window = lane_mask[wy0:wy1, wx0:wx1] > 0
            if lane_window.size:
                structure = np.ones((pad * 2 + 1, pad * 2 + 1), dtype=bool)
                near_window = ndimage.binary_dilation(lane_window, structure=structure)
                y_start = y_off - wy0
                x_start = x_off - wx0
                tss_in_or_near = near_window[y_start:y_start + h, x_start:x_start + w]
        else:
            tss_in_or_near = context.tss.lane_mask[y_off:y1, x_off:x1] > 0
        tss_in_or_near &= mask_bool

    pre["blocked_mask"] = blocked_mask
    pre["depth_penalty"] = depth_penalty
    pre["land_prox_penalty"] = land_prox_penalty
    pre["tss_in_or_near"] = tss_in_or_near
    pre["tss_direction"] = tss_direction

    return pre


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
