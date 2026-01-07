"""Route command backed by the corridor-limited A* implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ocean_router.api.dependencies import get_grid_spec
from ocean_router.core.geodesy import rhumb_distance_nm

app = typer.Typer(help="Compute a route using preprocessed caches")


@app.command()
def great_circle(
    start: str = typer.Argument(..., help="start lon,lat"),
    end: str = typer.Argument(..., help="end lon,lat"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save GeoJSON"),
) -> None:
    start_lon, start_lat = map(float, start.split(","))
    end_lon, end_lat = map(float, end.split(","))
    distance = rhumb_distance_nm(start_lon, start_lat, end_lon, end_lat)
    feature = {
        "type": "Feature",
        "properties": {"distance_nm": distance, "note": "rhumb line placeholder"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
        },
    }
    if output:
        output.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2))
        typer.echo(f"Saved route to {output}")
    else:
        typer.echo(json.dumps(feature, indent=2))


@app.command()
def info() -> None:
    grid = get_grid_spec()
    typer.echo(f"Grid CRS: {grid.crs}, size: {grid.width}x{grid.height}, dx: {grid.dx}")
