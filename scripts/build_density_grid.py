"""Normalize ship-density rasters to a float32 grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def normalize_density(density_tif: Path, grid: GridSpec, out_npy: Path) -> None:
    with rasterio.open(density_tif) as src:
        data = src.read(1, out_shape=(grid.height, grid.width), resampling=rasterio.enums.Resampling.bilinear)
    data = data.astype(np.float32)
    max_val = np.nanmax(data)
    if max_val > 0:
        data /= max_val
    data[np.isnan(data)] = 0
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    save_memmap(out_npy, data.astype(np.float32), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--density", type=Path, required=True, help="ship_density_1nm.tif")
    parser.add_argument("--out", type=Path, default=Path("data/processed/density/density_1nm.npy"))
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    normalize_density(args.density, grid, args.out)


if __name__ == "__main__":
    main()
