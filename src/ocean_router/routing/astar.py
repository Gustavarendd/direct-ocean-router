"""A* search constrained to a precomputed corridor mask."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ocean_router.core.geodesy import bearing_deg, rhumb_distance_nm
from ocean_router.core.grid import GridSpec
from ocean_router.routing.costs import CostContext, CostWeights
from ocean_router.routing.reconstruct import reconstruct_path


Move = Tuple[int, int]
MOVES: List[Move] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]


@dataclass
class AStarResult:
    path: List[Tuple[float, float]]
    explored: int
    cost: float
    success: bool


class CorridorAStar:
    def __init__(self, grid: GridSpec, corridor_mask: np.ndarray, x_off: int, y_off: int):
        self.grid = grid
        self.corridor_mask = corridor_mask
        self.x_off = x_off
        self.y_off = y_off

    def in_corridor(self, x: int, y: int) -> bool:
        cy = y - self.y_off
        cx = x - self.x_off
        if cy < 0 or cx < 0 or cy >= self.corridor_mask.shape[0] or cx >= self.corridor_mask.shape[1]:
            return False
        return bool(self.corridor_mask[cy, cx])

    def search(
        self,
        start_lonlat: Tuple[float, float],
        goal_lonlat: Tuple[float, float],
        context: CostContext,
        weights: CostWeights,
        min_depth: float,
        heuristic_weight: float = 1.0,
    ) -> AStarResult:
        start_x, start_y = self.grid.lonlat_to_xy(*start_lonlat)
        goal_x, goal_y = self.grid.lonlat_to_xy(*goal_lonlat)
        if not self.in_corridor(start_x, start_y) or not self.in_corridor(goal_x, goal_y):
            raise ValueError("start or goal outside corridor")

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, (start_x, start_y)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(start_x, start_y): 0.0}
        prev_bearing: Dict[Tuple[int, int], float] = {}
        explored = 0

        def heuristic(x: int, y: int) -> float:
            lon, lat = self.grid.xy_to_lonlat(x, y)
            return rhumb_distance_nm(lon, lat, *goal_lonlat)

        while open_set:
            _, current = heapq.heappop(open_set)
            explored += 1
            if current == (goal_x, goal_y):
                path_indices = reconstruct_path(came_from, current)
                lonlats = [self.grid.xy_to_lonlat(x, y) for x, y in path_indices]
                return AStarResult(path=lonlats, explored=explored, cost=g_score[current], success=True)

            cur_x, cur_y = current
            cur_lon, cur_lat = self.grid.xy_to_lonlat(cur_x, cur_y)
            for dx, dy in MOVES:
                nx, ny = cur_x + dx, cur_y + dy
                if not self.grid.valid_index(nx, ny):
                    continue
                if not self.in_corridor(nx, ny):
                    continue
                if context.blocked(ny, nx, min_depth):
                    continue
                move_bearing = bearing_deg(cur_lon, cur_lat, *self.grid.xy_to_lonlat(nx, ny))
                prev_b = prev_bearing.get(current)
                tentative_g = g_score[current] + context.move_cost(
                    ny, nx, cur_lat, prev_b, move_bearing, min_depth, weights,
                    prev_y=cur_y, prev_x=cur_x
                )
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    prev_bearing[neighbor] = move_bearing
                    f_score = tentative_g + heuristic(nx, ny) * heuristic_weight
                    heapq.heappush(open_set, (f_score, neighbor))
        return AStarResult(path=[], explored=explored, cost=float("inf"), success=False)
