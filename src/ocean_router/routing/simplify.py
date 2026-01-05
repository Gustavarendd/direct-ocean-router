"""Polyline simplification helpers."""
from __future__ import annotations

from typing import List, Tuple

from shapely.geometry import LineString


def douglas_peucker(points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    simplified = LineString(points).simplify(tolerance, preserve_topology=True)
    return list(simplified.coords)
