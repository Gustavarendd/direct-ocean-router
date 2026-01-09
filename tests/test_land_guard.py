from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
import fiona
from shapely.geometry import box, mapping

from ocean_router.core.grid import GridSpec
from ocean_router.land.land_guard import (
    LandGuardParams,
    build_land_index,
    route_with_land_guard,
    validate_route_no_land,
)


def test_strict_mask_preserves_narrow_channel() -> None:
    grid = GridSpec(crs="EPSG:4326", dx=1.0, dy=1.0, xmin=0.0, ymax=3.0, width=3, height=3)
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    shapes = [
        (box(0.0, 0.0, 1.49, 3.0), 1),
        (box(1.51, 0.0, 3.0, 3.0), 1),
    ]

    strict = features.rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    visual = features.rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    assert strict[:, 1].sum() == 0
    assert visual[:, 1].sum() > 0


class _StraightLineRouter:
    def route(self, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
        return [start, end]


class _GridRouter:
    def __init__(self, grid: GridSpec, land_mask: np.ndarray):
        self.grid = grid
        self.land_mask = land_mask

    def route(self, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
        start_x, start_y = self.grid.lonlat_to_xy(*start)
        goal_x, goal_y = self.grid.lonlat_to_xy(*end)
        if not self.grid.valid_index(start_x, start_y) or not self.grid.valid_index(goal_x, goal_y):
            return []
        if self.land_mask[start_y, start_x] or self.land_mask[goal_y, goal_x]:
            return []

        came_from = np.full((self.grid.height, self.grid.width, 2), -1, dtype=np.int32)
        queue: deque[tuple[int, int]] = deque()
        queue.append((start_x, start_y))
        came_from[start_y, start_x] = (start_x, start_y)

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        found = False
        while queue:
            x, y = queue.popleft()
            if (x, y) == (goal_x, goal_y):
                found = True
                break
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if not self.grid.valid_index(nx, ny):
                    continue
                if self.land_mask[ny, nx]:
                    continue
                if came_from[ny, nx, 0] != -1:
                    continue
                came_from[ny, nx] = (x, y)
                queue.append((nx, ny))

        if not found:
            return []

        path_xy: list[tuple[int, int]] = []
        cx, cy = goal_x, goal_y
        while True:
            path_xy.append((cx, cy))
            px, py = came_from[cy, cx]
            if (px, py) == (cx, cy):
                break
            cx, cy = int(px), int(py)

        path_xy.reverse()
        return [self.grid.xy_to_lonlat(x, y) for x, y in path_xy]


def _write_land_shapefile(path: Path, geom) -> None:
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(
        path,
        mode="w",
        driver="ESRI Shapefile",
        crs="EPSG:4326",
        schema=schema,
    ) as dst:
        dst.write({"geometry": mapping(geom), "properties": {"id": 1}})


def test_tiny_island_triggers_refinement(tmp_path: Path) -> None:
    island = box(0.49, 0.49, 0.51, 0.51)
    shp_path = tmp_path / "land.shp"
    _write_land_shapefile(shp_path, island)

    land_index = build_land_index(shp_path, cache_path=tmp_path / "land_index.pkl")

    def _local_factory(grid: GridSpec, land_mask: np.ndarray) -> _GridRouter:
        return _GridRouter(grid, land_mask)

    params = LandGuardParams(
        corridor_buffer_nm=5.0,
        refine_resolution_nm=0.5,
        max_refinements=3,
        local_cache_dir=tmp_path / "cache",
        local_router_factory=_local_factory,
    )

    start = (0.0, 0.5)
    end = (1.0, 0.5)
    route = route_with_land_guard(start, end, _StraightLineRouter(), land_index, params)
    report = validate_route_no_land(route, land_index)

    assert report.is_ok
    assert len(route) > 2
