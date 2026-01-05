"""Utilities to reconstruct paths from predecessor maps."""
from __future__ import annotations

from typing import Dict, List, Tuple


def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
