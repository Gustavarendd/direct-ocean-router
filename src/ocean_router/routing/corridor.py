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

from ocean_router.core.geodesy import bearing_deg, unwrap_lon
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
    buffer_deg = width_nm / 60.0
    span = grid.lon_span
    start_lon, start_lat = start
    end_lon, end_lat = end
    end_lon = unwrap_lon(end_lon, start_lon)

    lines = []
    if end_lon > grid.xmax:
        boundary = grid.xmax
        frac = (boundary - start_lon) / (end_lon - start_lon)
        lat_at = start_lat + (end_lat - start_lat) * frac
        lines.append(LineString([(start_lon, start_lat), (boundary, lat_at)]))
        lines.append(LineString([(grid.xmin, lat_at), (end_lon - span, end_lat)]))
    elif end_lon < grid.xmin:
        boundary = grid.xmin
        frac = (boundary - start_lon) / (end_lon - start_lon)
        lat_at = start_lat + (end_lat - start_lat) * frac
        lines.append(LineString([(start_lon, start_lat), (boundary, lat_at)]))
        lines.append(LineString([(grid.xmax, lat_at), (end_lon + span, end_lat)]))
    else:
        lines.append(LineString([(start_lon, start_lat), (end_lon, end_lat)]))

    corridor_geom = unary_union([line.buffer(buffer_deg, cap_style=1) for line in lines])
    
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


def build_corridor_from_path(
    grid: GridSpec,
    corridor_path: Sequence[Tuple[int, int]],
    land_mask: Optional[np.ndarray] = None,
    width_nm: float = 50.0,
    canals: Optional[Sequence["Canal"]] = None,
) -> Tuple[np.ndarray, int, int]:
    """Build a corridor mask around a backbone path (grid coordinates)."""
    if not corridor_path:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0

    cell_nm = max(grid.dx * 60.0, 1e-6)
    width_cells = max(2, int(round(width_nm / cell_nm)))
    xs = [p[0] for p in corridor_path]
    ys = [p[1] for p in corridor_path]

    padding = width_cells + 4
    x_min = max(0, min(xs) - padding)
    x_max = min(grid.width, max(xs) + padding + 1)
    y_min = max(0, min(ys) - padding)
    y_max = min(grid.height, max(ys) + padding + 1)

    h, w = y_max - y_min, x_max - x_min
    mask = np.zeros((h, w), dtype=np.uint8)

    for px, py in corridor_path:
        lx, ly = px - x_min, py - y_min
        y0 = max(0, ly - width_cells)
        y1 = min(h, ly + width_cells + 1)
        x0 = max(0, lx - width_cells)
        x1 = min(w, lx + width_cells + 1)
        mask[y0:y1, x0:x1] = 1

    if land_mask is not None:
        land_region = land_mask[y_min:y_max, x_min:x_max]
        mask = mask & (land_region == 0)

    if canals:
        canal_mask = canal_mask_window(canals, grid, x_min, y_min, h, w)
        if canal_mask.size:
            mask[canal_mask > 0] = 1

    return mask, x_min, y_min


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
    corridor_path: Optional[Sequence[Tuple[int, int]]] = None,
    corridor_backbone: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, int, int, Dict[str, Any]]:
    """Build corridor and precompute per-corridor arrays to speed up A*.

    Returns: (mask, x_off, y_off, precomputed_dict)
    precomputed_dict may contain:
      - blocked_mask: bool array
      - depth_penalty: float array
      - land_prox_penalty: float array
      - tss_in_or_near: bool array
      - tss_in_lane: bool array
      - tss_direction: float array (or -1)
      - tss_sepzone: bool array
      - tss_sepboundary: bool array
      - corridor_bearing: float array (or -1)
      - snap_corridor_mask: bool array
      - tss_correct_near: bool array
    """
    if corridor_backbone:
        mask, x_off, y_off = build_corridor_from_path(
            grid,
            corridor_backbone,
            land_mask=land_mask,
            width_nm=width_nm,
            canals=canals,
        )
    else:
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

    if corridor_path is None:
        sx, sy = grid.lonlat_to_xy(start[0], start[1])
        gx, gy = grid.lonlat_to_xy(end[0], end[1])
        corridor_path = [(sx, sy), (gx, gy)]

    sx, sy = grid.lonlat_to_xy(start[0], start[1])
    gx, gy = grid.lonlat_to_xy(end[0], end[1])
    sx_local = sx - x_off
    sy_local = sy - y_off
    gx_local = gx - x_off
    gy_local = gy - y_off
    if 0 <= sy_local < mask.shape[0] and 0 <= sx_local < mask.shape[1]:
        mask[sy_local, sx_local] = 1
    if 0 <= gy_local < mask.shape[0] and 0 <= gx_local < mask.shape[1]:
        mask[gy_local, gx_local] = 1

    if (
        context is not None
        and weights is not None
        and getattr(weights, "tss_lane_graph_lock_enabled", False)
        and context.tss is not None
        and context.tss.lane_graph is not None
        and mask.size
    ):
        from ocean_router.routing.lane_graph import lane_graph_mask_window

        cell_nm = max(grid.dx * 60.0, 1e-6)
        radius_cells = max(1, int(round(weights.tss_lane_graph_lock_radius_nm / cell_nm)))
        lane_graph_mask = lane_graph_mask_window(
            context.tss.lane_graph,
            grid,
            x_off,
            y_off,
            mask.shape[0],
            mask.shape[1],
            radius_cells,
        )
        if lane_graph_mask.any():
            y1 = y_off + mask.shape[0]
            x1 = x_off + mask.shape[1]
            lane_window = context.tss.lane_mask[y_off:y1, x_off:x1] > 0
            lock_zone = lane_window & (mask > 0)
            mask[lock_zone] = lane_graph_mask[lock_zone].astype(np.uint8)
            if 0 <= sy_local < mask.shape[0] and 0 <= sx_local < mask.shape[1]:
                mask[sy_local, sx_local] = 1
            if 0 <= gy_local < mask.shape[0] and 0 <= gx_local < mask.shape[1]:
                mask[gy_local, gx_local] = 1

    pre = precompute_corridor_arrays(
        mask,
        x_off,
        y_off,
        context=context,
        min_draft=min_draft,
        weights=weights,
        override_mask=canal_mask,
        corridor_path=corridor_path,
        grid=grid,
    )

    return mask, x_off, y_off, pre


def precompute_corridor_arrays(
    mask: np.ndarray,
    x_off: int,
    y_off: int,
    context: Optional[object] = None,
    min_draft: float = 10.0,
    weights: Optional[object] = None,
    tss_radius: Optional[int] = None,
    override_mask: Optional[np.ndarray] = None,
    corridor_path: Optional[Sequence[Tuple[int, int]]] = None,
    grid: Optional[GridSpec] = None,
) -> Dict[str, Any]:
    """Vectorized precompute of per-corridor arrays for faster A*."""
    pre: Dict[str, Any] = {}
    if context is None:
        return pre

    h, w = mask.shape
    if tss_radius is None and weights is not None:
        tss_radius = int(getattr(weights, "tss_proximity_check_radius", 2))
    if tss_radius is None:
        tss_radius = 2
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
    tss_in_lane = np.zeros((h, w), dtype=bool)
    tss_direction = np.full((h, w), -1.0, dtype=float)
    tss_sepzone = np.zeros((h, w), dtype=bool)
    tss_sepboundary = np.zeros((h, w), dtype=bool)
    if context.tss is not None:
        lane_window = context.tss.lane_mask[y_off:y1, x_off:x1] > 0
        tss_in_lane = lane_window & mask_bool
        tss_direction = context.tss.direction_field[y_off:y1, x_off:x1].astype(np.float32)
        if context.tss.sepzone_mask is not None:
            tss_sepzone = (context.tss.sepzone_mask[y_off:y1, x_off:x1] > 0) & mask_bool
        if context.tss.sepboundary_mask is not None:
            tss_sepboundary = (context.tss.sepboundary_mask[y_off:y1, x_off:x1] > 0) & mask_bool
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
    pre["tss_in_lane"] = tss_in_lane
    pre["tss_direction"] = tss_direction
    pre["tss_sepzone"] = tss_sepzone
    pre["tss_sepboundary"] = tss_sepboundary

    corridor_bearing: Optional[np.ndarray] = None
    if corridor_path and grid is not None and len(corridor_path) > 1:
        corridor_bearing = np.full((h, w), -1.0, dtype=np.float32)
        path_mask = np.ones((h, w), dtype=bool)
        bearings: list[float] = []
        for i, (px, py) in enumerate(corridor_path):
            if i == len(corridor_path) - 1:
                p0x, p0y = corridor_path[i - 1]
                p1x, p1y = px, py
            else:
                p0x, p0y = px, py
                p1x, p1y = corridor_path[i + 1]
            lon0, lat0 = grid.xy_to_lonlat(p0x, p0y)
            lon1, lat1 = grid.xy_to_lonlat(p1x, p1y)
            bearings.append(bearing_deg(lon0, lat0, lon1, lat1))

        for (px, py), brng in zip(corridor_path, bearings):
            lx = px - x_off
            ly = py - y_off
            if 0 <= ly < h and 0 <= lx < w:
                corridor_bearing[ly, lx] = brng
                path_mask[ly, lx] = False

        if not path_mask.all():
            _, indices = ndimage.distance_transform_edt(path_mask, return_indices=True)
            idx_y, idx_x = indices
            corridor_bearing = corridor_bearing[idx_y, idx_x]
            corridor_bearing[~mask_bool] = -1.0
            pre["corridor_bearing"] = corridor_bearing

    if (
        corridor_bearing is not None
        and context.tss is not None
        and weights is not None
        and getattr(weights, "tss_snap_corridor_enabled", False)
        and grid is not None
    ):
        lane_window = tss_in_lane
        valid = (corridor_bearing >= 0) & (tss_direction >= 0)
        angle = np.abs((tss_direction.astype(np.float32) - corridor_bearing + 180.0) % 360.0 - 180.0)
        correct_lane = lane_window & valid & (angle <= weights.tss_max_lane_deviation_deg)
        wrong_lane = lane_window & valid & (angle > weights.tss_max_lane_deviation_deg)
        if correct_lane.any():
            cell_nm = max(min(grid.dx, grid.dy) * 60.0, 1e-6)
            radius_cells = max(1, int(round(weights.tss_snap_corridor_radius_nm / cell_nm)))
            dist_correct = ndimage.distance_transform_edt(~correct_lane)
            tss_band = dist_correct <= radius_cells
            pre["tss_correct_near"] = tss_band
            snap_mask = mask_bool.copy()
            snap_band = tss_band.copy()
            if wrong_lane.any():
                dist_wrong = ndimage.distance_transform_edt(~wrong_lane)
                snap_band &= dist_correct <= dist_wrong
            snap_band &= ~wrong_lane
            snap_band &= mask_bool
            snap_mask[tss_band] = snap_band[tss_band]
            if corridor_path:
                path_local: list[tuple[int, int] | None] = []
                for px, py in corridor_path:
                    lx, ly = px - x_off, py - y_off
                    if 0 <= ly < h and 0 <= lx < w:
                        path_local.append((lx, ly))
                    else:
                        path_local.append(None)

                hit_indices = [
                    i for i, loc in enumerate(path_local)
                    if loc is not None and snap_mask[loc[1], loc[0]]
                ]
                if hit_indices:
                    first_hit = hit_indices[0]
                    last_hit = hit_indices[-1]
                    for i in range(0, first_hit + 1):
                        loc = path_local[i]
                        if (
                            loc is not None
                            and mask_bool[loc[1], loc[0]]
                            and not wrong_lane[loc[1], loc[0]]
                        ):
                            snap_mask[loc[1], loc[0]] = True
                    for i in range(last_hit, len(path_local)):
                        loc = path_local[i]
                        if (
                            loc is not None
                            and mask_bool[loc[1], loc[0]]
                            and not wrong_lane[loc[1], loc[0]]
                        ):
                            snap_mask[loc[1], loc[0]] = True
                    snap_mask &= mask_bool
                    start_loc = path_local[0]
                    end_loc = path_local[-1]
                    if (
                        start_loc is not None
                        and end_loc is not None
                        and snap_mask[start_loc[1], start_loc[0]]
                        and snap_mask[end_loc[1], end_loc[0]]
                    ):
                        pre["snap_corridor_mask"] = snap_mask

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
