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
from ocean_router.routing.corridor import precompute_corridor_arrays


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
    def _tss_check_nearby_wrong_way(
        lane_mask: np.ndarray,
        direction_field: np.ndarray,
        y: int,
        x: int,
        move_bearing: float,
        radius: int,
    ) -> bool:
        height, width = lane_mask.shape
        closest_wrong_way_dist = 1e9
        closest_right_way_dist = 1e9
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not lane_mask[ny, nx]:
                    continue
                dist = math.sqrt(float(dy * dy + dx * dx))
                if dist > radius:
                    continue
                preferred = float(direction_field[ny, nx])
                angle = _angle_diff(move_bearing, preferred)
                if angle > 90.0:
                    if dist < closest_wrong_way_dist:
                        closest_wrong_way_dist = dist
                else:
                    if dist < closest_right_way_dist:
                        closest_right_way_dist = dist
        if closest_wrong_way_dist < 1e9:
            if closest_right_way_dist >= closest_wrong_way_dist:
                return True
        return False

    @njit(cache=True)
    def _tss_alignment_along_path(
        lane_mask: np.ndarray,
        direction_field: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int,
        move_bearing: float,
        wrong_way_penalty: float,
        alignment_weight: float,
        goal_bearing: float,
        max_lane_deviation_deg: float,
        proximity_check_radius: int,
    ) -> float:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x = x0
        y = y0
        max_penalty = 0.0
        min_bonus = 0.0
        cells_in_lane = 0

        while True:
            if 0 <= y < lane_mask.shape[0] and 0 <= x < lane_mask.shape[1]:
                if lane_mask[y, x]:
                    cells_in_lane += 1
                    preferred = float(direction_field[y, x])
                    if goal_bearing >= 0.0:
                        lane_to_goal_angle = _angle_diff(preferred, goal_bearing)
                        if lane_to_goal_angle > 45.0:
                            return wrong_way_penalty * 0.5
                    angle = _angle_diff(move_bearing, preferred)
                    if angle > 90.0:
                        return wrong_way_penalty
                    bonus = -alignment_weight * (1.0 - angle / 90.0)
                    if bonus < min_bonus:
                        min_bonus = bonus
                else:
                    if _tss_check_nearby_wrong_way(
                        lane_mask, direction_field, y, x, move_bearing, proximity_check_radius
                    ):
                        return wrong_way_penalty * 0.5

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        if cells_in_lane > 0:
            return min_bonus
        return max_penalty

    @njit(cache=True)
    def _line_crosses_boundary(
        sepzone_mask: np.ndarray,
        sepboundary_mask: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int,
    ) -> bool:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x = x0
        y = y0
        prev_in_sepzone = sepzone_mask[y, x]
        prev_on_sepboundary = sepboundary_mask[y, x]

        while True:
            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            if 0 <= y < sepzone_mask.shape[0] and 0 <= x < sepzone_mask.shape[1]:
                curr_in_sepzone = sepzone_mask[y, x]
                if curr_in_sepzone != prev_in_sepzone:
                    return True
                prev_in_sepzone = curr_in_sepzone
                curr_on_sepboundary = sepboundary_mask[y, x]
                if (not prev_on_sepboundary) and curr_on_sepboundary:
                    return True
                prev_on_sepboundary = curr_on_sepboundary
        return False

    @njit(cache=True)
    def _line_crosses_sepboundary(
        sepboundary_mask: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int,
    ) -> bool:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x = x0
        y = y0
        while True:
            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            if 0 <= y < sepboundary_mask.shape[0] and 0 <= x < sepboundary_mask.shape[1]:
                if sepboundary_mask[y, x]:
                    return True
        return False

    @njit(cache=True)
    def _tss_boundary_crossing_penalty(
        lane_mask: np.ndarray,
        sepzone_mask: np.ndarray,
        sepboundary_mask: np.ndarray,
        prev_y: int,
        prev_x: int,
        y: int,
        x: int,
        lane_crossing_penalty: float,
        sepzone_crossing_penalty: float,
        sepboundary_crossing_penalty: float,
    ) -> float:
        penalty = 0.0
        was_in_lane = lane_mask[prev_y, prev_x]
        now_in_lane = lane_mask[y, x]
        if was_in_lane and not now_in_lane:
            penalty += lane_crossing_penalty

        was_in_sepzone = sepzone_mask[prev_y, prev_x]
        now_in_sepzone = sepzone_mask[y, x]
        if not was_in_sepzone and now_in_sepzone:
            penalty += sepzone_crossing_penalty
        elif was_in_sepzone and not now_in_sepzone:
            penalty += sepzone_crossing_penalty * 0.25

        if _line_crosses_boundary(sepzone_mask, sepboundary_mask, prev_y, prev_x, y, x):
            penalty += sepzone_crossing_penalty

        if _line_crosses_sepboundary(sepboundary_mask, prev_y, prev_x, y, x):
            penalty += sepboundary_crossing_penalty

        return penalty

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
        blocked_mask: np.ndarray,
        depth_penalty: np.ndarray,
        row_step_nms: np.ndarray,
        depth: np.ndarray,
        tss_in_or_near: np.ndarray,
        tss_in_lane: np.ndarray,
        tss_direction: np.ndarray,
        tss_sepzone: np.ndarray,
        tss_sepboundary: np.ndarray,
        tss_correct_near: np.ndarray,
        corridor_bearing: np.ndarray,
        tss_enabled: bool,
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
        tss_wrong_way_penalty: float,
        tss_alignment_weight: float,
        tss_off_lane_penalty: float,
        tss_lane_crossing_penalty: float,
        tss_sepzone_crossing_penalty: float,
        tss_sepboundary_crossing_penalty: float,
        tss_max_lane_deviation_deg: float,
        tss_proximity_check_radius: int,
        goal_x_g: int,
        goal_y_g: int,
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
        moves_bearing = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=np.float32)
        
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
            gcx = x_off + cx
            gcy = y_off + cy
            
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
                
                if blocked_mask.size > 0:
                    if blocked_mask[ny, nx]:
                        continue
                elif _blocked_check(depth, gny, gnx, min_draft, depth_nodata):
                    continue
                if tss_enabled and tss_sepzone.size > 0 and tss_sepzone[ny, nx]:
                    continue
                
                move_bearing = float(moves_bearing[m])
                step_nm = row_step_nms[cy, m]
                
                # Penalties
                if depth_penalty.size > 0:
                    penalty = depth_penalty[ny, nx]
                else:
                    penalty = _depth_penalty(depth, gny, gnx, min_draft, near_shore_penalty, depth_nodata)

                if tss_enabled:
                    in_near = False
                    prev_in_near = False
                    if tss_in_or_near.size > 0:
                        in_near = tss_in_or_near[ny, nx]
                        prev_in_near = tss_in_or_near[cy, cx]
                    if tss_off_lane_penalty > 0.0 and (in_near or prev_in_near):
                        in_lane = False
                        if tss_in_lane.size > 0:
                            in_lane = tss_in_lane[ny, nx]
                        if not in_lane:
                            if tss_correct_near.size > 0 and tss_correct_near[ny, nx]:
                                penalty += tss_off_lane_penalty

                    if in_near or prev_in_near:
                        goal_bearing = -1.0
                        if corridor_bearing.size > 0:
                            cb = corridor_bearing[ny, nx]
                            if cb >= 0:
                                goal_bearing = cb
                        if goal_bearing < 0.0:
                            goal_bearing = (math.degrees(math.atan2(goal_x_g - gnx, -(goal_y_g - gny))) + 360.0) % 360.0
                        penalty += _tss_alignment_along_path(
                            tss_in_lane,
                            tss_direction,
                            gcy,
                            gcx,
                            gny,
                            gnx,
                            move_bearing,
                            tss_wrong_way_penalty,
                            tss_alignment_weight,
                            goal_bearing,
                            tss_max_lane_deviation_deg,
                            tss_proximity_check_radius,
                        )

                    penalty += _tss_boundary_crossing_penalty(
                        tss_in_lane,
                        tss_sepzone,
                        tss_sepboundary,
                        gcy,
                        gcx,
                        gny,
                        gnx,
                        tss_lane_crossing_penalty,
                        tss_sepzone_crossing_penalty,
                        tss_sepboundary_crossing_penalty,
                    )
                
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

        pre = precompute_corridor_arrays(
            self.corridor_mask,
            self.x_off,
            self.y_off,
            context=context,
            min_draft=min_depth,
            weights=weights,
            corridor_path=[(start_x, start_y), (goal_x, goal_y)],
            grid=self.grid,
        )
        blocked_mask = pre.get("blocked_mask", np.zeros_like(self.corridor_mask, dtype=bool))
        depth_penalty = pre.get("depth_penalty", np.zeros_like(self.corridor_mask, dtype=np.float32))
        tss_in_or_near = pre.get("tss_in_or_near", np.zeros_like(self.corridor_mask, dtype=bool))
        tss_in_lane = pre.get("tss_in_lane", np.zeros_like(self.corridor_mask, dtype=bool))
        tss_direction = pre.get("tss_direction", np.full(self.corridor_mask.shape, -1.0, dtype=np.float32))
        tss_sepzone = pre.get("tss_sepzone", np.zeros_like(self.corridor_mask, dtype=bool))
        tss_sepboundary = pre.get("tss_sepboundary", np.zeros_like(self.corridor_mask, dtype=bool))
        tss_correct_near = pre.get("tss_correct_near", np.zeros_like(self.corridor_mask, dtype=bool))
        corridor_bearing = pre.get("corridor_bearing", np.full(self.corridor_mask.shape, -1.0, dtype=np.float32))

        rows = np.arange(h, dtype=np.float32)
        lat = self.grid.ymax - (self.y_off + rows + 0.5) * self.grid.dy
        lat_factor = np.cos(np.deg2rad(lat))
        grid_dx_nm = self.grid.dx * 60.0
        grid_dy_nm = self.grid.dy * 60.0
        base_step = np.hypot(grid_dx_nm * lat_factor, grid_dy_nm) / math.sqrt(2.0)
        row_step_nms = (base_step[:, None] * np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)[None, :]).astype(np.float32)
        
        # Run Numba core
        came_from_x, came_from_y, g_score, final_cost, explored, success = _astar_numba_core(
            self.corridor_mask,
            blocked_mask,
            depth_penalty,
            row_step_nms,
            depth,
            tss_in_or_near,
            tss_in_lane,
            tss_direction,
            tss_sepzone,
            tss_sepboundary,
            tss_correct_near,
            corridor_bearing,
            context.tss is not None,
            sx, sy, gx, gy,
            self.x_off, self.y_off,
            min_depth, nodata,
            self.grid.dx, self.grid.dy,
            self.grid.xmin, self.grid.ymax,
            goal_lonlat[0], goal_lonlat[1],
            weights.near_shore_depth_penalty,
            weights.turn_penalty_weight,
            weights.tss_wrong_way_penalty,
            weights.tss_alignment_weight,
            weights.tss_off_lane_penalty,
            weights.tss_lane_crossing_penalty,
            weights.tss_sepzone_crossing_penalty,
            weights.tss_sepboundary_crossing_penalty,
            weights.tss_max_lane_deviation_deg,
            int(weights.tss_proximity_check_radius),
            goal_x,
            goal_y
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
