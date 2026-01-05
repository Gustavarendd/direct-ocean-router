"""Rasterize TSS lanes, separation zones, and direction fields."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import fiona
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points
from rasterio import features
import rasterio

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def rasterize_mask(shapes_iter: Iterable[Tuple[object, int]], grid: GridSpec, out_path: Path) -> np.ndarray:
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    mask = features.rasterize(shapes_iter, out_shape=(grid.height, grid.width), transform=transform, fill=0, dtype="uint8")
    save_memmap(out_path, mask.astype(np.uint8), dtype=np.uint8)
    return mask


def build_direction_field(centerlines: list[LineString], grid: GridSpec, out_path: Path, influence_nm: float = 2.0) -> None:
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    field = np.full((grid.height, grid.width), -1, dtype=np.int16)
    influence_deg = influence_nm / 60
    xs = np.linspace(grid.xmin, grid.xmax, grid.width, endpoint=False) + grid.dx / 2
    ys = np.linspace(grid.ymax, grid.ymin, grid.height, endpoint=False) + (-grid.dy / 2)
    for iy, lat in enumerate(ys):
        for ix, lon in enumerate(xs):
            point = (lon, lat)
            nearest_bearing = None
            for line in centerlines:
                if line.distance(LineString([point, point])) > influence_deg:
                    continue
                nearest_pt = nearest_points(line, LineString([point, point]))[0]
                if nearest_pt.is_empty:
                    continue
                segment = list(line.coords)
                # naive: use first segment bearing
                if len(segment) >= 2:
                    lon1, lat1 = segment[0]
                    lon2, lat2 = segment[1]
                    bearing = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360
                    nearest_bearing = bearing
                    break
            if nearest_bearing is not None:
                field[iy, ix] = int(nearest_bearing)
    save_memmap(out_path, field, dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--lanes", type=Path, required=True, help="TSS lane polygons")
    parser.add_argument("--sepzones", type=Path, help="Separation zone polygons")
    parser.add_argument("--centerlines", type=Path, required=True, help="TSS centerlines")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/tss"))
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    args.outdir.mkdir(parents=True, exist_ok=True)

    with fiona.open(args.lanes) as lane_src:
        lane_shapes = [(shape(f["geometry"]), 1) for f in lane_src]
    lane_mask = rasterize_mask(lane_shapes, grid, args.outdir / "tss_lane_mask_1nm.npy")

    sep_mask = None
    if args.sepzones:
        with fiona.open(args.sepzones) as sep_src:
            sep_shapes = [(shape(f["geometry"]), 1) for f in sep_src]
        sep_mask = rasterize_mask(sep_shapes, grid, args.outdir / "tss_sepzone_mask_1nm.npy")

    with fiona.open(args.centerlines) as center_src:
        centerlines = [LineString(shape(f["geometry"])) for f in center_src]
    build_direction_field(centerlines, grid, args.outdir / "tss_dir_field_1nm.npy")

    if sep_mask is not None:
        lane_mask[sep_mask.astype(bool)] = 0
        save_memmap(args.outdir / "tss_lane_mask_1nm.npy", lane_mask, dtype=np.uint8)


if __name__ == "__main__":
    main()
