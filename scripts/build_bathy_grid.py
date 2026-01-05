"""Resample ETOPO bathymetry onto the routing grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def resample_bathy(etopo_path: Path, grid: GridSpec, out_npy: Path, nodata: int = -32768) -> None:
    with rasterio.open(etopo_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            grid.crs,
            src.width,
            src.height,
            left=grid.xmin,
            bottom=grid.ymin,
            right=grid.xmax,
            top=grid.ymax,
            dst_width=grid.width,
            dst_height=grid.height,
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": grid.crs, "transform": transform, "width": width, "height": height, "nodata": nodata})
        data = np.full((grid.height, grid.width), nodata, dtype=np.int16)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=grid.crs,
            resampling=Resampling.bilinear,
        )
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    save_memmap(out_npy, data, dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--etopo", type=Path, required=True, help="Path to ETOPO 2022 raster")
    parser.add_argument("--out", type=Path, default=Path("data/processed/bathy/depth_1nm.npy"))
    parser.add_argument("--nodata", type=int, default=-32768)
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    resample_bathy(args.etopo, grid, args.out, nodata=args.nodata)


if __name__ == "__main__":
    main()
