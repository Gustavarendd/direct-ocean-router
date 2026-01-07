"""Canal override helpers for routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ocean_router.core.grid import GridSpec


@dataclass(frozen=True)
class Canal:
    id: str
    name: str
    width_nm: float
    waypoints: List[Tuple[float, float]]  # (lon, lat)


def load_canals(path: Path) -> List[Canal]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    canals = []
    for item in data.get("canals", []):
        waypoints = [(float(lon), float(lat)) for lon, lat in item.get("waypoints", [])]
        if len(waypoints) < 2:
            continue
        canals.append(
            Canal(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                width_nm=float(item.get("width_nm", 2.0)),
                waypoints=waypoints,
            )
        )
    return canals


def _radius_cells(width_nm: float, cell_nm: float) -> int:
    if cell_nm <= 0:
        return 1
    return max(1, int(round(width_nm / cell_nm / 2.0)))


def _draw_thick_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int) -> None:
    """Draw a thick line on a 2D mask using Bresenham with a square brush."""
    h, w = mask.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if 0 <= x < w and 0 <= y < h:
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)
            mask[y_min:y_max, x_min:x_max] = 1

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def canal_mask_window(
    canals: Optional[Sequence[Canal]],
    grid: GridSpec,
    x_off: int,
    y_off: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Rasterize canals into a corridor-local mask."""
    if not canals or h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cell_nm = grid.dx * 60.0

    for canal in canals:
        radius = _radius_cells(canal.width_nm, cell_nm)
        coords = [grid.lonlat_to_xy(lon, lat) for lon, lat in canal.waypoints]
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            _draw_thick_line(mask, x0 - x_off, y0 - y_off, x1 - x_off, y1 - y_off, radius)

    return mask


def canal_mask_coarse(
    canals: Optional[Sequence[Canal]],
    grid: GridSpec,
    scale: int,
) -> Optional[np.ndarray]:
    """Rasterize canals onto a coarse grid for macro routing."""
    if not canals or scale <= 0:
        return None

    h, w = grid.height, grid.width
    pad_h = (scale - h % scale) % scale
    pad_w = (scale - w % scale) % scale
    ch = (h + pad_h) // scale
    cw = (w + pad_w) // scale

    mask = np.zeros((ch, cw), dtype=np.uint8)
    coarse_cell_nm = grid.dx * 60.0 * scale

    for canal in canals:
        radius = _radius_cells(canal.width_nm, coarse_cell_nm)
        coords = [grid.lonlat_to_xy(lon, lat) for lon, lat in canal.waypoints]
        coarse_coords = [(x // scale, y // scale) for x, y in coords]
        for (x0, y0), (x1, y1) in zip(coarse_coords, coarse_coords[1:]):
            _draw_thick_line(mask, x0, y0, x1, y1, radius)

    return mask
