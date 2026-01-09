"""Preprocessing commands wired to scripts stubs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ocean_router.core.grid import GridSpec

app = typer.Typer(help="Preprocessing utilities to rasterize and cache data")


@app.command()
def all() -> None:
    typer.echo("Run scripts/build_land_mask.py, build_bathy_grid.py, build_tss_fields.py, and build_density_grid.py with your data.")


@app.command()
def land() -> None:
    typer.echo("Use scripts/build_land_mask.py to rasterize OSM land polygons to the routing grid.")


@app.command()
def bathy() -> None:
    typer.echo("Use scripts/build_bathy_grid.py to resample ETOPO onto the routing grid.")


@app.command()
def tss() -> None:
    typer.echo("Use scripts/build_tss_fields.py to rasterize lanes, separation zones, and direction fields.")


@app.command("tile-rasters")
def tile_rasters(
    grid: Path = typer.Option(..., help="Grid JSON (e.g. configs/grid_1nm.json)"),
    land: Optional[Path] = typer.Option(None, help="Path to land mask .npy"),
    land_buffered: Optional[Path] = typer.Option(None, help="Optional buffered land mask .npy"),
    bathy: Optional[Path] = typer.Option(None, help="Path to bathy depth .npy"),
    bathy_nodata: int = typer.Option(-32768, help="Nodata value for bathy tiles"),
    tile_size: int = typer.Option(2048, help="Tile size in pixels"),
    outdir: Path = typer.Option(Path("data/processed"), help="Output base directory"),
) -> None:
    """Tile large raster arrays into tile-backed memmaps with metadata."""
    grid_spec = GridSpec.from_file(grid)
    suffix = grid.stem.replace("grid_", "")
    if land:
        _tile_raster(land, outdir / "land_tiles" / suffix, grid_spec, tile_size, nodata=0)
    if land_buffered:
        _tile_raster(land_buffered, outdir / "land_tiles" / f"{suffix}_buffered", grid_spec, tile_size, nodata=0)
    if bathy:
        _tile_raster(bathy, outdir / "bathy_tiles" / suffix, grid_spec, tile_size, nodata=bathy_nodata)


def _tile_raster(path: Path, tiles_dir: Path, grid: GridSpec, tile_size: int, nodata: int | float) -> None:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(path, mmap_mode="r")
    height, width = array.shape
    tiles_x = int(np.ceil(width / tile_size))
    tiles_y = int(np.ceil(height / tile_size))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            tile_path = tiles_dir / f"tile_{tx}_{ty}.npy"
            if tile_path.exists():
                continue
            tile = np.array(array[y0:y1, x0:x1])
            np.save(tile_path, tile)

    meta_path = tiles_dir / "meta.json"
    payload = {
        "xmin": grid.xmin,
        "ymax": grid.ymax,
        "dx": grid.dx,
        "dy": grid.dy,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "dtype": str(array.dtype),
        "nodata": nodata,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tiles_dir": str(tiles_dir),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
