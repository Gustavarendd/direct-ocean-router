"""Rasterize OSM land polygons to a global 1 nm grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
import fiona

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def rasterize_land(polygons_path: Path, grid: GridSpec, out_tif: Path, out_npy: Path, iterations: int = 1) -> None:
    with fiona.open(polygons_path) as src:
        shapes = [shape(feat["geometry"]) for feat in src]
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    raster = features.rasterize(
        ((geom, 1) for geom in shapes),
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype="uint8",
        crs=grid.crs,
        transform=transform,
    ) as dst:
        dst.write(raster, 1)
    save_memmap(out_npy, raster.astype(np.uint8), dtype=np.uint8)

    if iterations > 0:
        from scipy.ndimage import binary_dilation

        buffered = binary_dilation(raster, iterations=iterations).astype(np.uint8)
        save_memmap(out_npy.with_name(out_npy.stem + "_buffered.npy"), buffered, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--land", type=Path, required=True, help="Path to land_polygons.shp")
    parser.add_argument("--out", type=Path, default=Path("data/processed/land/land_mask_1nm.tif"))
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    out_tif = args.out
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out_npy = out_tif.with_suffix(".npy")

    rasterize_land(args.land, grid, out_tif, out_npy, iterations=2)


if __name__ == "__main__":
    main()
