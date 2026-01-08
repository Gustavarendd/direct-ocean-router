"""Typer CLI for preprocessing and routing."""
from __future__ import annotations

import os
import typer

from ocean_router.cli import route_cmd, preprocess_cmd

app = typer.Typer(help="Offline preprocessing and corridor-limited routing")
app.add_typer(route_cmd.app, name="route")
app.add_typer(preprocess_cmd.app, name="preprocess")


@app.command()
def resolution(
    res: str = typer.Argument(None, help="Resolution to set (1nm, 0.5nm, or 0.25nm). Omit to show current."),
) -> None:
    """Show or set the grid resolution for routing.
    
    Examples:
        ocean-router resolution           # Show current
        ocean-router resolution 0.5nm     # Set to 0.5nm
        ocean-router resolution 1nm       # Set to 1nm
        ocean-router resolution 0.25nm    # Set to 0.25nm
    """
    from ocean_router.api.dependencies import get_resolution, set_resolution, VALID_RESOLUTIONS
    
    if res is None:
        current = os.environ.get("OCEAN_ROUTER_RESOLUTION", "1nm")
        typer.echo(f"Current resolution: {current}")
        typer.echo(f"Valid options: {', '.join(sorted(VALID_RESOLUTIONS))}")
        typer.echo("")
        typer.echo("To change resolution, either:")
        typer.echo(f"  1. Run: ocean-router resolution 0.5nm")
        typer.echo(f"  2. Set env: export OCEAN_ROUTER_RESOLUTION=0.5nm")
    elif res in VALID_RESOLUTIONS:
        set_resolution(res)
        typer.echo(f"Resolution set to: {res}")
        typer.echo("Note: This only affects the current session.")
        typer.echo(f"For persistent change, use: export OCEAN_ROUTER_RESOLUTION={res}")
    else:
        typer.echo(f"Invalid resolution '{res}'. Valid options: {', '.join(sorted(VALID_RESOLUTIONS))}")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show information about the current configuration and available data."""
    from pathlib import Path
    from ocean_router.api.dependencies import (
        get_resolution, get_grid_spec, get_bathy, get_land_mask, get_tss, _project_root
    )
    
    typer.echo("=== Ocean Router Configuration ===")
    typer.echo(f"Resolution: {get_resolution()}")
    
    grid = get_grid_spec()
    typer.echo(f"Grid: {grid.width}x{grid.height} cells, dx={grid.dx:.8f}°")
    
    typer.echo("")
    typer.echo("=== Data Availability ===")
    
    bathy = get_bathy()
    typer.echo(f"Bathymetry: {'✓ loaded' if bathy else '✗ not found'}")
    
    land = get_land_mask()
    typer.echo(f"Land mask:  {'✓ loaded' if land else '✗ not found'}")
    
    tss = get_tss()
    typer.echo(f"TSS fields: {'✓ loaded' if tss else '✗ not found'}")
    
    # Show available data files
    typer.echo("")
    typer.echo("=== Available Data Files ===")
    data_dir = _project_root() / "data" / "processed"
    for subdir in ["bathy", "land", "tss"]:
        path = data_dir / subdir
        if path.exists():
            files = list(path.glob("*.npy"))
            for f in sorted(files):
                size_mb = f.stat().st_size / (1024 * 1024)
                typer.echo(f"  {subdir}/{f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    app()
