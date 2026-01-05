"""Sanity checks for processed masks."""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from ocean_router.core.grid import GridSpec
from ocean_router.data.land import LandMask
from ocean_router.data.bathy import load_bathy


def sample_checks(grid: GridSpec, land: LandMask, bathy_path: Path, samples: int = 10) -> list[str]:
    notes: list[str] = []
    bathy = load_bathy(bathy_path)
    for _ in range(samples):
        x = random.randint(0, grid.width - 1)
        y = random.randint(0, grid.height - 1)
        lon, lat = grid.xy_to_lonlat(x, y)
        if land.buffered[y, x]:
            notes.append(f"Buffered land at ({lon:.2f},{lat:.2f}) blocks routing.")
        depth = bathy.depth[y, x]
        if depth == bathy.nodata:
            notes.append(f"Nodata depth at ({lon:.2f},{lat:.2f}).")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--land", type=Path, default=Path("data/processed/land/land_mask_1nm.npy"))
    parser.add_argument("--bathy", type=Path, default=Path("data/processed/bathy/depth_1nm.npy"))
    parser.add_argument("--buffered", type=Path, default=Path("data/processed/land/land_mask_buffered_1nm.npy"))
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    land = LandMask(args.land, buffered_path=args.buffered)
    notes = sample_checks(grid, land, args.bathy, samples=args.samples)
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
