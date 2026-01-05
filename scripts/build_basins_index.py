"""Optionally build a serialized basin index for fast lookups."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basins", type=Path, default=Path("configs/basins.geojson"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/basins/basins_index.pkl"))
    args = parser.parse_args()

    gdf = gpd.read_file(args.basins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(gdf, f)


if __name__ == "__main__":
    main()
