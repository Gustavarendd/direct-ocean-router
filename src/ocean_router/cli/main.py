"""Typer CLI for preprocessing and routing."""
from __future__ import annotations

import typer

from ocean_router.cli import route_cmd, preprocess_cmd

app = typer.Typer(help="Offline preprocessing and corridor-limited routing")
app.add_typer(route_cmd.app, name="route")
app.add_typer(preprocess_cmd.app, name="preprocess")


if __name__ == "__main__":
    app()
