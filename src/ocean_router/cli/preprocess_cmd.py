"""Preprocessing commands wired to scripts stubs."""
from __future__ import annotations

import typer

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
