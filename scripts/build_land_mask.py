"""Rasterize OSM land polygons to a global 1 nm grid."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
import fiona

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


def rasterize_land(
    polygons_path: Path,
    grid: GridSpec,
    out_tif: Path,
    out_npy: Path,
    *,
    all_touched: bool,
    buffer_iters: int = 0,
) -> None:
    """Rasterize land polygons using streaming to avoid memory issues."""
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    
    print(f"Reading land polygons from {polygons_path}...")
    
    # Stream polygons directly to rasterizer without loading all into memory
    def shape_generator():
        with fiona.open(polygons_path) as src:
            total = len(src)
            print(f"  Total features: {total:,}")
            for i, feat in enumerate(src):
                if i % 50000 == 0:
                    print(f"  Processing feature {i:,}/{total:,}...")
                yield (shape(feat["geometry"]), 1)
    
    print("Rasterizing...")
    raster = features.rasterize(
        shape_generator(),
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    
    print(f"Land pixels: {np.sum(raster > 0):,} / {raster.size:,}")
    
    print(f"Saving GeoTIFF to {out_tif}...")
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
        compress="lzw",
    ) as dst:
        dst.write(raster, 1)
    
    print(f"Saving numpy array to {out_npy}...")
    save_memmap(out_npy, raster.astype(np.uint8), dtype=np.uint8)

    if buffer_iters > 0:
        from scipy.ndimage import binary_dilation

        buffered = binary_dilation(raster, iterations=buffer_iters).astype(np.uint8)
        save_memmap(out_npy.with_name(out_npy.stem + "_buffered.npy"), buffered, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument(
        "--resolution",
        type=str,
        default="1nm",
        help="Resolution suffix for output files (1nm or 0.5nm)",
    )
    parser.add_argument("--land", type=Path, required=True, help="Path to land_polygons.shp")
    parser.add_argument("--out", type=Path, default=None, help="Path to save output (auto-generated if not provided)")
    parser.add_argument(
        "--purpose",
        choices=["strict", "visual"],
        default="strict",
        help="Mask purpose for naming and defaults (strict for routing, visual for display)",
    )
    parser.add_argument(
        "--all-touched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rasterize with all_touched=True/False (defaults based on --purpose)",
    )
    parser.add_argument(
        "--buffer-iters",
        type=int,
        default=0,
        help="Optional binary dilation iterations to buffer land (0 disables)",
    )
    args = parser.parse_args()
    
    # Derive file suffix from resolution (e.g., "0.5nm" -> "05nm")
    suffix = args.resolution.replace(".", "")
    
    all_touched = args.all_touched if args.all_touched is not None else args.purpose == "visual"

    # Auto-generate output path if not provided
    if args.out is None:
        out_tif = Path(f"data/processed/land/land_mask_{args.purpose}_{suffix}.tif")
    else:
        out_tif = args.out
    
    grid = GridSpec.from_file(args.grid)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out_npy = out_tif.with_suffix(".npy")

    print(f"[RESOLUTION] Building land mask at {args.resolution} ({grid.width}x{grid.height})")
    rasterize_land(
        args.land,
        grid,
        out_tif,
        out_npy,
        all_touched=all_touched,
        buffer_iters=args.buffer_iters,
    )


if __name__ == "__main__":
    main()
