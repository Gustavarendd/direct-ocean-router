"""Numba-accelerated A* search for maximum performance."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ocean_router.core.grid import GridSpec
from ocean_router.routing.costs import CostContext, CostWeights


@dataclass
class AStarResult:
    path: List[Tuple[float, float]]
    explored: int
    cost: float
    success: bool


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _blocked_check(depth: np.ndarray, y: int, x: int, min_draft: float, nodata: int) -> bool:
        """Check if cell is blocked (land or too shallow)."""
        d = depth[y, x]
        if d == nodata:
            return True
        # GEBCO: negative = underwater. Safe if depth <= -min_draft
        return d > -min_draft

    @njit(cache=True)
    def _depth_penalty(depth: np.ndarray, y: int, x: int, min_draft: float, penalty_weight: float, nodata: int) -> float:
        """Calculate penalty for shallow water."""
        d = depth[y, x]
        if d == nodata or d > -min_draft:
            return 1e30
        water_depth = -d
        slack = max(water_depth - min_draft, 0.001)
        return penalty_weight / slack

    @njit(cache=True)
    def _angle_diff(a: float, b: float) -> float:
        diff = (a - b + 180.0) % 360.0 - 180.0
        return abs(diff)

    @njit(cache=True)
    def _heuristic_nm(x: int, y: int, goal_x: int, goal_y: int, 
                     xmin: float, ymax: float, dx: float, dy: float,
                     goal_lon: float, goal_lat: float) -> float:
        """Fast approximate distance heuristic in NM."""
        lon = xmin + (x + 0.5) * dx
        lat = ymax - (y + 0.5) * dy
        dlat = abs(lat - goal_lat) * 60.0
        mid_lat = (lat + goal_lat) / 2.0
        dlon = abs(lon - goal_lon) * 60.0 * math.cos(math.radians(mid_lat))
        return math.sqrt(dlat * dlat + dlon * dlon)

    @njit(cache=True, parallel=False)
    def _astar_numba_core(
        corridor: np.ndarray,
        depth: np.ndarray,
        start_x: int, start_y: int,
        goal_x: int, goal_y: int,
        x_off: int, y_off: int,
        min_draft: float,
        depth_nodata: int,
        grid_dx: float, grid_dy: float,
        xmin: float, ymax: float,
        goal_lon: float, goal_lat: float,
        near_shore_penalty: float,
        turn_penalty: float,
        max_iterations: int = 5000000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
        """
        Core A* loop in Numba.
        Returns (came_from_x, came_from_y, g_score, final_cost, explored, success).
        """
        h, w = corridor.shape
        INF = 1e30
        
        # Arrays for tracking
        g_score = np.full((h, w), INF, dtype=np.float32)
        came_from_x = np.full((h, w), -1, dtype=np.int32)
        came_from_y = np.full((h, w), -1, dtype=np.int32)
        prev_bearing = np.full((h, w), -1.0, dtype=np.float32)
        closed = np.zeros((h, w), dtype=np.uint8)
        
        g_score[start_y, start_x] = 0.0
        
        # Moves: (dx, dy, cost_mult)
        moves_dx = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
        moves_dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
        moves_mult = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)
        
        # Simple heap simulation with arrays (max size estimate)
        heap_size = min(h * w, 2000000)
        heap_f = np.zeros(heap_size, dtype=np.float32)
        heap_x = np.zeros(heap_size, dtype=np.int32)
        heap_y = np.zeros(heap_size, dtype=np.int32)
        heap_len = 1
        
        # Push start
        init_h = _heuristic_nm(start_x + x_off, start_y + y_off, goal_x + x_off, goal_y + y_off,
                               xmin, ymax, grid_dx, grid_dy, goal_lon, goal_lat)
        heap_f[0] = init_h
        heap_x[0] = start_x
        heap_y[0] = start_y
        
        explored = 0
        
        while heap_len > 0 and explored < max_iterations:
            # Pop minimum
            min_idx = 0
            for i in range(1, heap_len):
                if heap_f[i] < heap_f[min_idx]:
                    min_idx = i
            
            cx = heap_x[min_idx]
            cy = heap_y[min_idx]
            
            # Remove by swapping with last
            heap_len -= 1
            if min_idx < heap_len:
                heap_f[min_idx] = heap_f[heap_len]
                heap_x[min_idx] = heap_x[heap_len]
                heap_y[min_idx] = heap_y[heap_len]
            
            if closed[cy, cx]:
                continue
            closed[cy, cx] = 1
            explored += 1
            
            if cx == goal_x and cy == goal_y:
                return came_from_x, came_from_y, g_score, g_score[goal_y, goal_x], explored, True
            
            cur_g = g_score[cy, cx]
            cur_bearing = prev_bearing[cy, cx]
            
            # Current position for lat calculation
            cur_lat = ymax - (cy + y_off + 0.5) * grid_dy
            lat_factor = math.cos(math.radians(cur_lat))
            
            for m in range(8):
                dx_m = moves_dx[m]
                dy_m = moves_dy[m]
                mult = moves_mult[m]
                
                nx = cx + dx_m
                ny = cy + dy_m
                
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if closed[ny, nx]:
                    continue
                if not corridor[ny, nx]:
                    continue
                
                # Global coords for depth check
                gnx = x_off + nx
                gny = y_off + ny
                
                if _blocked_check(depth, gny, gnx, min_draft, depth_nodata):
                    continue
                
                # Move bearing (approximate)
                move_bearing = (math.degrees(math.atan2(float(dx_m), float(-dy_m))) + 360.0) % 360.0
                
                # Step cost in NM
                base_step = math.sqrt((grid_dx * 60.0 * lat_factor)**2 + (grid_dy * 60.0)**2) / 1.414
                step_nm = mult * base_step
                
                # Penalties
                penalty = _depth_penalty(depth, gny, gnx, min_draft, near_shore_penalty, depth_nodata)
                
                if cur_bearing >= 0:
                    angle_diff = _angle_diff(move_bearing, cur_bearing)
                    penalty += turn_penalty * angle_diff / 180.0
                
                tentative_g = cur_g + step_nm + penalty
                
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    came_from_x[ny, nx] = cx
                    came_from_y[ny, nx] = cy
                    prev_bearing[ny, nx] = move_bearing
                    
                    h_val = _heuristic_nm(nx + x_off, ny + y_off, goal_x + x_off, goal_y + y_off,
                                         xmin, ymax, grid_dx, grid_dy, goal_lon, goal_lat)
                    f_val = tentative_g + h_val
                    
                    # Push to heap
                    if heap_len < heap_size:
                        heap_f[heap_len] = f_val
                        heap_x[heap_len] = nx
                        heap_y[heap_len] = ny
                        heap_len += 1
        
        return came_from_x, came_from_y, g_score, INF, explored, False


class NumbaAStar:
    """Numba-accelerated A* for maximum performance on long routes."""
    
    def __init__(self, grid: GridSpec, corridor_mask: np.ndarray, x_off: int, y_off: int):
        self.grid = grid
        self.corridor_mask = corridor_mask.astype(np.uint8)
        self.x_off = x_off
        self.y_off = y_off
    
    def search(
        self,
        start_lonlat: Tuple[float, float],
        goal_lonlat: Tuple[float, float],
        context: CostContext,
        weights: CostWeights,
        min_depth: float,
    ) -> AStarResult:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba not installed")
        
        start_x, start_y = self.grid.lonlat_to_xy(*start_lonlat)
        goal_x, goal_y = self.grid.lonlat_to_xy(*goal_lonlat)
        
        sx, sy = start_x - self.x_off, start_y - self.y_off
        gx, gy = goal_x - self.x_off, goal_y - self.y_off
        
        h, w = self.corridor_mask.shape
        if not (0 <= sy < h and 0 <= sx < w and self.corridor_mask[sy, sx]):
            raise ValueError("start outside corridor")
        if not (0 <= gy < h and 0 <= gx < w and self.corridor_mask[gy, gx]):
            raise ValueError("goal outside corridor")
        
        # Get depth array
        depth = context.bathy.depth
        nodata = int(context.bathy.nodata)
        
        # Run Numba core
        came_from_x, came_from_y, g_score, final_cost, explored, success = _astar_numba_core(
            self.corridor_mask,
            depth,
            sx, sy, gx, gy,
            self.x_off, self.y_off,
            min_depth, nodata,
            self.grid.dx, self.grid.dy,
            self.grid.xmin, self.grid.ymax,
            goal_lonlat[0], goal_lonlat[1],
            weights.near_shore_depth_penalty,
            weights.turn_penalty_weight
        )
        
        if not success:
            return AStarResult(path=[], explored=explored, cost=float('inf'), success=False)
        
        # Reconstruct path
        path_indices = []
        x, y = gx, gy
        while x >= 0 and y >= 0:
            path_indices.append((x + self.x_off, y + self.y_off))
            px, py = came_from_x[y, x], came_from_y[y, x]
            x, y = px, py
        path_indices.reverse()
        
        lonlats = [self.grid.xy_to_lonlat(x, y) for x, y in path_indices]
        return AStarResult(
            path=lonlats,
            explored=explored,
            cost=float(final_cost),
            success=True
        )
