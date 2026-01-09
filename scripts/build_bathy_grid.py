"""Resample GEBCO bathymetry tiles onto the routing grid."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from typing import List

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def merge_gebco_tiles(tile_paths: List[Path]) -> rasterio.DatasetReader:
    """Merge multiple GEBCO tiles into a single virtual dataset."""
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(datasets)
    
    # Get metadata from first dataset
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
    })
    
    # Close source datasets
    for ds in datasets:
        ds.close()
    
    # Write to memory file and return
    memfile = MemoryFile()
    with memfile.open(**out_meta) as mem_dst:
        mem_dst.write(mosaic)
    
    return memfile.open()


def resample_bathy(gebco_src, grid: GridSpec, out_npy: Path, nodata: int = -32768) -> None:
    """Resample bathymetry data onto the routing grid."""
    transform = rasterio.transform.from_bounds(
        grid.xmin, grid.ymin, grid.xmax, grid.ymax,
        grid.width, grid.height
    )
    
    data = np.full((grid.height, grid.width), nodata, dtype=np.int16)
    reproject(
        source=rasterio.band(gebco_src, 1),
        destination=data,
        src_transform=gebco_src.transform,
        src_crs=gebco_src.crs,
        dst_transform=transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    save_memmap(out_npy, data, dtype=np.int16)
    print(f"Saved bathymetry grid to {out_npy}")
    print(f"  Shape: {data.shape}, min: {data.min()}, max: {data.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bathymetry grid from GEBCO tiles")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--resolution", type=str, default="1nm",
                        help="Resolution suffix for output files (1nm or 0.5nm)")
    parser.add_argument("--gebco-dir", type=Path, required=True, 
                        help="Directory containing GEBCO GeoTIFF tiles")
    parser.add_argument("--out", type=Path, default=None, help="Output path (auto-generated if not provided)")
    parser.add_argument("--nodata", type=int, default=-32768)
    args = parser.parse_args()

    # Derive file suffix from resolution (e.g., "0.5nm" -> "05nm")
    suffix = args.resolution.replace(".", "")
    
    # Auto-generate output path if not provided
    if args.out is None:
        out_path = Path(f"data/processed/bathy/depth_{suffix}.npy")
    else:
        out_path = args.out

    grid = GridSpec.from_file(args.grid)
    
    # Find all GEBCO tiles
    tile_paths = list(args.gebco_dir.glob("gebco_*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No GEBCO tiles found in {args.gebco_dir}")
    
    print(f"[RESOLUTION] Building bathymetry at {args.resolution} ({grid.width}x{grid.height})")
    print(f"Found {len(tile_paths)} GEBCO tiles:")
    for p in tile_paths:
        print(f"  - {p.name}")
    
    print("Merging tiles...")
    merged_src = merge_gebco_tiles(tile_paths)
    
    print(f"Resampling to grid ({grid.width}x{grid.height})...")
    resample_bathy(merged_src, grid, out_path, nodata=args.nodata)
    merged_src.close()


if __name__ == "__main__":
    main()
