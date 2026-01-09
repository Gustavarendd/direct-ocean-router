"""Optimized A* search with hierarchical coarse-to-fine approach."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, TYPE_CHECKING

import numpy as np
from scipy import ndimage

from ocean_router.core.geodesy import bearing_deg, rhumb_distance_nm, shortest_dlon
from ocean_router.core.grid import GridSpec
from ocean_router.routing.costs import CostContext, CostWeights
from ocean_router.routing.corridor import precompute_corridor_arrays

if TYPE_CHECKING:
    from ocean_router.data.bathy import Bathy
    from ocean_router.data.land import LandMask


class _ArrayBathy:
    def __init__(self, depth: np.ndarray, nodata: int = -32768) -> None:
        self._depth = depth
        self.nodata = nodata

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    def depth_window(self, y_off: int, x_off: int, height: int, width: int) -> np.ndarray:
        return self._depth[y_off:y_off + height, x_off:x_off + width]

    def depth_value(self, y: int, x: int) -> float:
        return float(self._depth[y, x])


class _ArrayLandMask:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def sample(self, y: int, x: int) -> int:
        return int(self._mask[y, x])

    def window(self, y_off: int, x_off: int, height: int, width: int) -> np.ndarray:
        return self._mask[y_off:y_off + height, x_off:x_off + width]


@dataclass
class AStarResult:
    path: List[Tuple[float, float]]
    explored: int
    cost: float
    success: bool


# Pre-compute moves and their costs (dx, dy, cost_multiplier)
MOVES_8 = np.array([
    (0, -1, 1.0),      # N
    (1, -1, 1.414),    # NE
    (1, 0, 1.0),       # E
    (1, 1, 1.414),     # SE
    (0, 1, 1.0),       # S
    (-1, 1, 1.414),    # SW
    (-1, 0, 1.0),      # W
    (-1, -1, 1.414),   # NW
], dtype=np.float32)

MOVE_BEARINGS = np.array([
    (math.degrees(math.atan2(dx, -dy)) % 360)
    for dx, dy, _ in MOVES_8
], dtype=np.float32)


def _precompute_row_step_nms(grid: GridSpec, y_off: int, h: int) -> np.ndarray:
    """Precompute step distances for each row in NM for MOVES_8."""
    grid_dx_nm = grid.dx * 60.0
    grid_dy_nm = grid.dy * 60.0
    rows = np.arange(h, dtype=np.float32)
    lat = grid.ymax - (y_off + rows + 0.5) * grid.dy
    lat_factor = np.cos(np.deg2rad(lat))
    base_step = np.hypot(grid_dx_nm * lat_factor, grid_dy_nm) / math.sqrt(2.0)
    return (base_step[:, None] * MOVES_8[:, 2][None, :]).astype(np.float32)


def _precompute_goal_bearing(x_off: int, y_off: int, w: int, h: int, goal_x: int, goal_y: int) -> np.ndarray:
    """Precompute grid-based bearing from each cell to the goal."""
    xs = (x_off + np.arange(w, dtype=np.float32))[None, :]
    ys = (y_off + np.arange(h, dtype=np.float32))[:, None]
    dx = goal_x - xs
    dy = goal_y - ys
    bearing = np.degrees(np.arctan2(dx, -dy)) % 360.0
    return bearing.astype(np.float32)


def _line_of_sight_clear(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    is_blocked: Callable[[int, int], bool]
) -> bool:
    """Check if there's a clear line of sight between two grid points using Bresenham."""
    x0, y0 = p0
    x1, y1 = p1
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        if is_blocked(x, y):
            return False
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return True


def _line_too_close_to_land(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    distance_from_land: np.ndarray,
    min_distance_cells: int = 5,
    tss: Optional['TSSFields'] = None
) -> bool:
    """Check if any point along a line is too close to land.
    
    Uses Bresenham's algorithm to check all cells along the line.
    Points inside TSS lanes are exempt from the check (shipping lanes are safe).
    
    Args:
        p0: Start point (x, y)
        p1: End point (x, y)
        distance_from_land: Array of distances from land (in cells)
        min_distance_cells: Minimum allowed distance from land (default 5 = ~5nm)
        tss: Optional TSSFields - if provided, points in TSS lanes are exempt
        
    Returns:
        True if line passes too close to land (outside of TSS lanes)
    """
    x0, y0 = p0
    x1, y1 = p1
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    h, w = distance_from_land.shape
    
    while True:
        # Check bounds and distance from land
        if 0 <= y < h and 0 <= x < w:
            dist = distance_from_land[y, x]
            if dist < min_distance_cells:
                # Allow if inside a TSS lane (shipping lanes are safe near land)
                if tss is not None and tss.in_lane(y, x):
                    pass  # OK - in a shipping lane
                else:
                    return True  # Too close to land outside shipping lane
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return False


def smooth_path_xy(
    path_xy: List[Tuple[int, int]],
    is_blocked: Callable[[int, int], bool],
    max_skip: int = 500
) -> List[Tuple[int, int]]:
    """Smooth a grid path using line-of-sight checks (string pulling).
    
    First checks if direct start-to-end is possible, then does iterative
    smoothing until no more improvements can be made.
    """
    if len(path_xy) < 3:
        return path_xy
    
    def line_validator(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
        return _line_of_sight_clear(p0, p1, is_blocked)
    
    return _smooth_path_with_validator(path_xy, line_validator, max_skip)


def _smooth_path_with_validator(
    path_xy: List[Tuple[int, int]],
    line_is_valid: Callable[[Tuple[int, int], Tuple[int, int]], bool],
    max_skip: int = 500
) -> List[Tuple[int, int]]:
    """Smooth a grid path using a custom line validator.
    
    Args:
        path_xy: List of (x, y) grid coordinates
        line_is_valid: Function that takes (start, end) points and returns True if the line is valid
        max_skip: Maximum number of points to try skipping at once
        
    Returns:
        Smoothed path
    """
    if len(path_xy) < 3:
        return path_xy
    
    # First check: can we go directly from start to end?
    if line_is_valid(path_xy[0], path_xy[-1]):
        return [path_xy[0], path_xy[-1]]
    
    # Do multiple passes until no improvement
    current = path_xy
    while True:
        smoothed = _smooth_pass_with_validator(current, line_is_valid, max_skip)
        if len(smoothed) >= len(current):
            break  # No improvement
        current = smoothed
    
    return current


def _smooth_pass_with_validator(
    path_xy: List[Tuple[int, int]],
    line_is_valid: Callable[[Tuple[int, int], Tuple[int, int]], bool],
    max_skip: int
) -> List[Tuple[int, int]]:
    """Single pass of path smoothing with custom validator."""
    if len(path_xy) < 3:
        return path_xy
    
    smoothed = [path_xy[0]]
    i = 0
    
    while i < len(path_xy) - 1:
        # Try to skip as far as possible (check farthest first for efficiency)
        best_j = i + 1
        
        # For long distances, check end first
        if line_is_valid(path_xy[i], path_xy[-1]):
            smoothed.append(path_xy[-1])
            break
        
        # Binary search for farthest visible point
        lo, hi = i + 1, min(i + max_skip, len(path_xy) - 1)
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_is_valid(path_xy[i], path_xy[mid]):
                lo = mid
            else:
                hi = mid - 1
        
        smoothed.append(path_xy[lo])
        i = lo
    
    return smoothed


def _smooth_pass(
    path_xy: List[Tuple[int, int]],
    is_blocked: Callable[[int, int], bool],
    max_skip: int
) -> List[Tuple[int, int]]:
    """Single pass of path smoothing."""
    if len(path_xy) < 3:
        return path_xy
    
    smoothed = [path_xy[0]]
    i = 0
    
    while i < len(path_xy) - 1:
        # Try to skip as far as possible (check farthest first for efficiency)
        best_j = i + 1
        
        # For long distances, check end first
        if _line_of_sight_clear(path_xy[i], path_xy[-1], is_blocked):
            smoothed.append(path_xy[-1])
            break
        
        # Binary search for farthest visible point
        lo, hi = i + 1, min(i + max_skip, len(path_xy) - 1)
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _line_of_sight_clear(path_xy[i], path_xy[mid], is_blocked):
                lo = mid
            else:
                hi = mid - 1
        
        smoothed.append(path_xy[lo])
        i = lo
    
    return smoothed


def _build_coarse_traversable(
    corridor_mask: np.ndarray,
    bathy_depth: np.ndarray,
    min_draft: float,
    x_off: int,
    y_off: int,
    scale: int
) -> np.ndarray:
    """Build a coarse-resolution traversable mask using max-pooling approach."""
    h, w = corridor_mask.shape
    
    # Pad to make divisible by scale
    pad_h = (scale - h % scale) % scale
    pad_w = (scale - w % scale) % scale
    
    if pad_h > 0 or pad_w > 0:
        corridor_padded = np.pad(corridor_mask, ((0, pad_h), (0, pad_w)), constant_values=0)
    else:
        corridor_padded = corridor_mask
    
    ch, cw = corridor_padded.shape[0] // scale, corridor_padded.shape[1] // scale
    
    # Reshape and check if any cell in each block is traversable
    corridor_blocks = corridor_padded.reshape(ch, scale, cw, scale)
    coarse_corridor = corridor_blocks.any(axis=(1, 3)).astype(np.uint8)
    
    # Now check bathymetry - extract the global region
    gy0, gx0 = y_off, x_off
    gy1, gx1 = y_off + corridor_padded.shape[0], x_off + corridor_padded.shape[1]
    
    # Bounds check and extract
    bh, bw = bathy_depth.shape
    if gy1 > bh or gx1 > bw or gy0 < 0 or gx0 < 0:
        # Partial overlap - use slower method
        for cy in range(ch):
            for cx in range(cw):
                if not coarse_corridor[cy, cx]:
                    continue
                y0, y1 = gy0 + cy * scale, gy0 + (cy + 1) * scale
                x0, x1 = gx0 + cx * scale, gx0 + (cx + 1) * scale
                y0, y1 = max(0, y0), min(bh, y1)
                x0, x1 = max(0, x0), min(bw, x1)
                if y1 <= y0 or x1 <= x0:
                    coarse_corridor[cy, cx] = 0
                    continue
                block_depth = bathy_depth[y0:y1, x0:x1]
                if not (block_depth <= -min_draft).any():
                    coarse_corridor[cy, cx] = 0
    else:
        # Fast path - full overlap
        depth_region = bathy_depth[gy0:gy1, gx0:gx1]
        depth_blocks = depth_region.reshape(ch, scale, cw, scale)
        # A block is traversable if ANY cell has sufficient depth
        depth_ok = (depth_blocks <= -min_draft).any(axis=(1, 3))
        coarse_corridor &= depth_ok.astype(np.uint8)
    
    return coarse_corridor


def _coarse_astar(
    coarse_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    scale: int
) -> Optional[List[Tuple[int, int]]]:
    """Fast A* on coarse grid to find approximate path."""
    cs = (start[0] // scale, start[1] // scale)
    cg = (goal[0] // scale, goal[1] // scale)
    
    h, w = coarse_mask.shape
    
    # Bounds check
    if not (0 <= cs[1] < h and 0 <= cs[0] < w):
        return None
    if not (0 <= cg[1] < h and 0 <= cg[0] < w):
        return None
    
    def heuristic(x: int, y: int) -> float:
        dx = abs(x - cg[0])
        dx = min(dx, w - dx)
        return math.sqrt(dx**2 + (y - cg[1])**2) * scale
    
    open_set = [(0.0, cs)]
    came_from = {}
    g_score = {cs: 0.0}
    closed = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed:
            continue
        closed.add(current)
        
        if current == cg:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return [(x * scale + scale // 2, y * scale + scale // 2) for x, y in reversed(path)]
        
        cx, cy = current
        for dx, dy, mult in MOVES_8:
            nx, ny = int(cx + dx), int(cy + dy)
            if not (0 <= ny < h):
                continue
            nx %= w
            if not coarse_mask[ny, nx]:
                continue
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
                
            tentative_g = g_score[current] + mult * scale
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_set, (f, neighbor))
    
    return None


def _expand_coarse_path_to_corridor(
    coarse_path: List[Tuple[int, int]],
    corridor_mask: np.ndarray,
    scale: int,
    margin: int = 2
) -> np.ndarray:
    """Create a narrow corridor around the coarse path."""
    h, w = corridor_mask.shape
    narrow = np.zeros((h, w), dtype=np.uint8)
    
    # Mark cells near the coarse path. Cap expansion relative to scale so
    # extremely large scales do not create huge corridors.
    expansion = max(2, scale // 4 + margin)
    for cx, cy in coarse_path:
        # Convert to fine coordinates
        y0 = max(0, cy - expansion)
        y1 = min(h, cy + expansion + 1)
        x0 = max(0, cx - expansion)
        x1 = min(w, cx + expansion + 1)
        narrow[y0:y1, x0:x1] = 1
    
    # Intersect with original corridor and traversability
    return narrow & corridor_mask


class HierarchicalAStar:
    """Two-level hierarchical A* for faster long-distance routing."""
    
    def __init__(
        self,
        grid: GridSpec,
        corridor_mask: np.ndarray,
        x_off: int,
        y_off: int,
        coarse_scale: int = 8
    ):
        self.grid = grid
        self.corridor_mask = corridor_mask
        self.x_off = x_off
        self.y_off = y_off
        self.coarse_scale = coarse_scale
        
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
        
        # Convert to corridor-local coordinates
        sx, sy = start_x - self.x_off, start_y - self.y_off
        gx, gy = goal_x - self.x_off, goal_y - self.y_off
        
        corridor_mask = self.precomputed.get("snap_corridor_mask") if hasattr(self, "precomputed") else None
        if corridor_mask is None or not corridor_mask.size:
            corridor_mask = self.corridor_mask
        h, w = corridor_mask.shape
        wrap_x = self.x_off == 0 and w == self.grid.width
        if not (0 <= sy < h and 0 <= sx < w and corridor_mask[sy, sx]):
            raise ValueError("start outside corridor")
        if not (0 <= gy < h and 0 <= gx < w and corridor_mask[gy, gx]):
            raise ValueError("goal outside corridor")
        
        # Build coarse traversable mask
        coarse_mask = _build_coarse_traversable(
            self.corridor_mask,
            context.bathy.depth,
            min_depth,
            self.x_off,
            self.y_off,
            self.coarse_scale
        )
        
        # Run coarse A*
        coarse_path = _coarse_astar(coarse_mask, (sx, sy), (gx, gy), self.coarse_scale)
        
        if coarse_path is None:
            return AStarResult(path=[], explored=0, cost=float('inf'), success=False)
        
        # Build narrow corridor around coarse path
        # Use a tighter expansion around the coarse path to keep the fine
        # search corridor small for long-distance routes.
        narrow_corridor = _expand_coarse_path_to_corridor(
            coarse_path, self.corridor_mask, self.coarse_scale, margin=1
        )
        
        # Run fine-grained A* in the narrow corridor
        return self._fine_astar(
            narrow_corridor, (sx, sy), (gx, gy),
            context, weights, min_depth, goal_lonlat, heuristic_weight=heuristic_weight
        )
    
    def _fine_astar(
        self,
        corridor: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        context: CostContext,
        weights: CostWeights,
        min_depth: float,
        goal_lonlat: Tuple[float, float],
        heuristic_weight: float = 1.0,
    ) -> AStarResult:
        """Fine-grained A* search in a narrow corridor."""
        h, w = corridor.shape
        corridor_mask = corridor
        wrap_x = self.x_off == 0 and w == self.grid.width
        
        # Use arrays for faster access
        INF = 1e30
        g_score = np.full((h, w), INF, dtype=np.float32)
        g_score[start[1], start[0]] = 0.0
        
        came_from_x = np.full((h, w), -1, dtype=np.int32)
        came_from_y = np.full((h, w), -1, dtype=np.int32)
        prev_bearing = np.full((h, w), -1.0, dtype=np.float32)
        
        closed = np.zeros((h, w), dtype=np.uint8)
        
        # Heuristic: straight-line distance in NM (approximate)
        goal_lon, goal_lat = goal_lonlat

        # Precompute grid-derived constants to avoid repeated work in inner loop
        grid_dx_nm = context.grid_dx * 60.0
        grid_dy_nm = context.grid_dy * 60.0
        moves_base = MOVES_8[:, 2]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        
        def heuristic(x: int, y: int) -> float:
            gx = self.x_off + x
            gy = self.y_off + y
            lon, lat = self.grid.xy_to_lonlat(gx, gy)
            # Approximate distance in NM
            dlat = abs(lat - goal_lat) * 60
            dlon = abs(shortest_dlon(lon, goal_lon)) * 60 * math.cos(math.radians((lat + goal_lat) / 2))
            return math.sqrt(dlat**2 + dlon**2)
        
        open_set = [(heuristic(start[0], start[1]), start)]
        explored = 0

        # Precompute blocked mask for corridor cells to avoid repeated context.blocked() calls
        blocked_mask = np.zeros_like(corridor, dtype=np.bool_)
        ys, xs = np.where(corridor)
        for y, x in zip(ys, xs):
            # convert to global coords for blocked check
            gnx = self.x_off + x
            gny = self.y_off + y
            blocked_mask[y, x] = context.blocked(gny, gnx, min_depth)
        
        # Timing counters for profiling
        time_heappop = 0.0
        time_neighbor_gen = 0.0
        time_blocked_check = 0.0
        time_cost_calc = 0.0
        time_heappush = 0.0
        # Disable frequent timing prints in tight loops for performance
        sample_interval = 10**9  # effectively never
        
        while open_set:
            t0 = time.perf_counter()
            _, (cx, cy) = heapq.heappop(open_set)
            time_heappop += time.perf_counter() - t0
            
            if closed[cy, cx]:
                continue
            closed[cy, cx] = 1
            explored += 1
            
            # Periodic timing report
            if explored % sample_interval == 0:
                total_time = time_heappop + time_neighbor_gen + time_blocked_check + time_cost_calc + time_heappush
                print(f"[A* TIMING @ {explored} nodes] heappop={time_heappop*1000:.1f}ms, neighbor={time_neighbor_gen*1000:.1f}ms, blocked={time_blocked_check*1000:.1f}ms, cost={time_cost_calc*1000:.1f}ms, heappush={time_heappush*1000:.1f}ms, total={total_time*1000:.1f}ms")
            
            if (cx, cy) == goal:
                # Reconstruct path
                path_indices = []
                x, y = goal
                while x >= 0 and y >= 0:
                    path_indices.append((x + self.x_off, y + self.y_off))
                    px, py = came_from_x[y, x], came_from_y[y, x]
                    x, y = px, py
                path_indices.reverse()
                
                # Smooth the path using line-of-sight (TSS-aware and land-aware)
                # Primary smoothing: conservative land clearance
                min_land_distance_cells = 5

                def is_blocked(x: int, y: int) -> bool:
                    return context.blocked(y, x, min_depth)

                def line_is_valid(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
                    if not _line_of_sight_clear(p0, p1, is_blocked):
                        return False
                    if context.land is not None:
                        if _line_too_close_to_land(p0, p1, context.land.distance_from_land, min_land_distance_cells, context.tss):
                            return False
                    if context.tss is not None:
                        x0, y0 = p0
                        x1, y1 = p1
                        if context.tss.line_crosses_boundary(y0, x0, y1, x1):
                            return False
                    return True

                smoothed = _smooth_path_with_validator(path_indices, line_is_valid)

                # Secondary (relaxed) smoothing: try again with reduced land clearance
                # This helps remove grid-aligned stepping when the straight
                # segment stays within TSS lanes or is otherwise acceptable.
                relaxed_min_land = max(1, min_land_distance_cells // 2)

                def line_is_valid_relaxed(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
                    if not _line_of_sight_clear(p0, p1, is_blocked):
                        return False
                    if context.land is not None:
                        # Allow closer passes if the line is mostly within TSS lanes
                        if _line_too_close_to_land(p0, p1, context.land.distance_from_land, relaxed_min_land, context.tss):
                            return False
                    if context.tss is not None:
                        x0, y0 = p0
                        x1, y1 = p1
                        if context.tss.line_crosses_boundary(y0, x0, y1, x1):
                            return False
                    return True

                # Attempt relaxed smoothing only if initial smoothing left many points
                if len(smoothed) > max(10, len(path_indices) // 4):
                    smoothed_relaxed = _smooth_path_with_validator(smoothed, line_is_valid_relaxed, max_skip=500)
                    # Keep the shorter (fewer points) result if valid
                    if len(smoothed_relaxed) < len(smoothed):
                        smoothed = smoothed_relaxed
                
                lonlats = [self.grid.xy_to_lonlat(x, y) for x, y in smoothed]
                return AStarResult(
                    path=lonlats,
                    explored=explored,
                    cost=float(g_score[goal[1], goal[0]]),
                    success=True
                )
            
            cur_g = g_score[cy, cx]
            cur_bearing = prev_bearing[cy, cx]
            
            # Get current position for cost calculation
            gx_cur = self.x_off + cx
            gy_cur = self.y_off + cy
            cur_lon, cur_lat = self.grid.xy_to_lonlat(gx_cur, gy_cur)

            # Precompute lat factor and step distances for 8 moves
            lat_factor = math.cos(math.radians(cur_lat))
            # per-move step in NM for this latitude (array of 8)
            # avoid repeated sqrt by multiplying base multipliers
            base_step = math.hypot(grid_dx_nm * lat_factor, grid_dy_nm)
            step_nms = (moves_base * base_step * inv_sqrt2).tolist()
            
            # Process each valid neighbor
            for idx, (dx, dy, base_mult) in enumerate(MOVES_8):
                dx, dy = int(dx), int(dy)
                nx, ny = cx + dx, cy + dy

                t0 = time.perf_counter()
                if not (0 <= ny < h):
                    continue
                if wrap_x:
                    nx %= w
                elif not (0 <= nx < w):
                    continue
                if closed[ny, nx]:
                    continue
                if not corridor_mask[ny, nx]:
                    continue
                time_neighbor_gen += time.perf_counter() - t0
                
                # Convert to global coordinates for blocked check
                gnx = self.x_off + nx
                gny = self.y_off + ny
                
                # Use precomputed blocked mask
                t0 = time.perf_counter()
                if blocked_mask[ny, nx]:
                    time_blocked_check += time.perf_counter() - t0
                    continue
                time_blocked_check += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                # Calculate move bearing (approximate)
                move_bearing = math.degrees(math.atan2(dx, -dy)) % 360

                # Calculate cost (use precomputed value)
                step_nm = step_nms[idx]
                
                # Simplified penalty calculation (use locals to avoid attribute lookups)
                penalty = 0.0
                penalty += context.bathy.depth_penalty(gny, gnx, min_depth, weights.near_shore_depth_penalty)

                if context.land:
                    penalty += context.land.proximity_penalty(gny, gnx, weights.land_proximity_penalty, max_distance_cells=weights.land_proximity_max_distance_cells)

                if context.tss:
                    gcx = self.x_off + cx
                    gcy = self.y_off + cy
                    in_near = context.tss.in_or_near_lane(gny, gnx, radius=weights.tss_proximity_check_radius)
                    if weights.tss_off_lane_penalty > 0 and in_near and not context.tss.in_lane(gny, gnx):
                        wrong_only = context.tss._check_nearby_wrong_way(
                            gny, gnx, move_bearing, radius=weights.tss_proximity_check_radius
                        )
                        if not wrong_only:
                            penalty += weights.tss_off_lane_penalty
                    if in_near:
                        goal_x_g, goal_y_g = self.x_off + goal[0], self.y_off + goal[1]
                        goal_bearing = math.degrees(math.atan2(goal_x_g - gcx, -(goal_y_g - gcy))) % 360
                    else:
                        goal_bearing = None
                    penalty += context.tss.alignment_penalty(
                        gny,
                        gnx,
                        move_bearing,
                        weights.tss_wrong_way_penalty,
                        weights.tss_alignment_weight,
                        prev_y=gcy,
                        prev_x=gcx,
                        goal_bearing=goal_bearing,
                        max_lane_deviation_deg=weights.tss_max_lane_deviation_deg,
                        proximity_check_radius=weights.tss_proximity_check_radius,
                        grid=context.grid,
                    )
                    penalty += context.tss.boundary_crossing_penalty(gcy, gcx, gny, gnx, weights.tss_lane_crossing_penalty, weights.tss_sepzone_crossing_penalty, weights.tss_sepboundary_crossing_penalty)
                
                if cur_bearing >= 0:
                    angle_diff = abs(((move_bearing - cur_bearing + 180) % 360) - 180)
                    penalty += weights.turn_penalty_weight * angle_diff / 180
                
                tentative_g = cur_g + step_nm + penalty
                time_cost_calc += time.perf_counter() - t0
                
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    came_from_x[ny, nx] = cx
                    came_from_y[ny, nx] = cy
                    prev_bearing[ny, nx] = move_bearing
                    f = tentative_g + heuristic(nx, ny) * heuristic_weight
                    t0 = time.perf_counter()
                    heapq.heappush(open_set, (f, (nx, ny)))
                    time_heappush += time.perf_counter() - t0
        
        # Final timing report
        total_time = time_heappop + time_neighbor_gen + time_blocked_check + time_cost_calc + time_heappush
        print(f"[A* FINAL TIMING] explored={explored}, heappop={time_heappop*1000:.1f}ms, neighbor={time_neighbor_gen*1000:.1f}ms, blocked={time_blocked_check*1000:.1f}ms, cost={time_cost_calc*1000:.1f}ms, heappush={time_heappush*1000:.1f}ms, total={total_time*1000:.1f}ms")
        
        return AStarResult(path=[], explored=explored, cost=float('inf'), success=False)


def _build_global_coarse_mask(
    bathy: "Bathy",
    land: Optional["LandMask"],
    min_draft: float,
    scale: int,
    override_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a coarse traversable mask for the entire grid using windowed reads."""
    h, w = bathy.depth.shape
    ch = int(math.ceil(h / scale))
    cw = int(math.ceil(w / scale))
    coarse_mask = np.zeros((ch, cw), dtype=np.uint8)
    threshold = 1

    for cy in range(ch):
        y0 = cy * scale
        y1 = min(h, (cy + 1) * scale)
        for cx in range(cw):
            x0 = cx * scale
            x1 = min(w, (cx + 1) * scale)
            depth_block = bathy.depth_window(y0, x0, y1 - y0, x1 - x0)
            depth_ok = (depth_block <= -min_draft).any()
            if not depth_ok:
                continue
            if land is not None:
                land_block = land.window(y0, x0, y1 - y0, x1 - x0)
                water_ok = (land_block == 0).sum() >= threshold
                if not water_ok:
                    continue
            coarse_mask[cy, cx] = 1

    if override_mask is not None and override_mask.shape == coarse_mask.shape:
        coarse_mask |= override_mask.astype(np.uint8)

    return coarse_mask


def _global_coarse_astar(
    coarse_mask: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    scale: int,
    grid_dx: float,
    grid_dy: float,
    ymax: float,
    bathy: Optional["Bathy"] = None,
    land: Optional["LandMask"] = None,
    min_draft: float = 10.0
) -> Optional[List[Tuple[int, int]]]:
    """A* on coarse global grid. Returns path in fine grid coordinates.
    
    If start/goal coarse cell is not traversable (e.g., coastal), we check if
    the actual fine-scale start/goal point is water and allow it anyway.
    """
    sx, sy = start_xy[0] // scale, start_xy[1] // scale
    gx, gy = goal_xy[0] // scale, goal_xy[1] // scale
    
    h, w = coarse_mask.shape
    
    # Make a copy of coarse mask so we can modify it for start/goal
    local_mask = coarse_mask.copy()
    
    # Check if fine-scale start point is water, force-enable coarse cell
    if 0 <= sy < h and 0 <= sx < w and not local_mask[sy, sx]:
        # Check fine scale
        fx, fy = start_xy
        fine_ok = True
        if bathy is not None:
            fine_ok = fine_ok and bathy.depth_value(fy, fx) <= -min_draft
        if land is not None:
            fine_ok = fine_ok and not land.sample(fy, fx)
        if fine_ok:
            local_mask[sy, sx] = 1  # Force enable
    
    # Check if fine-scale goal point is water, force-enable coarse cell  
    if 0 <= gy < h and 0 <= gx < w and not local_mask[gy, gx]:
        fx, fy = goal_xy
        fine_ok = True
        if bathy is not None:
            fine_ok = fine_ok and bathy.depth_value(fy, fx) <= -min_draft
        if land is not None:
            fine_ok = fine_ok and not land.sample(fy, fx)
        if fine_ok:
            local_mask[gy, gx] = 1  # Force enable
    
    if not (0 <= sy < h and 0 <= sx < w and local_mask[sy, sx]):
        return None
    if not (0 <= gy < h and 0 <= gx < w and local_mask[gy, gx]):
        return None
    
    # Heuristic in approximate NM
    def heuristic(cx: int, cy: int) -> float:
        # Convert coarse coords to approximate lat/lon
        lat = ymax - (cy * scale + scale // 2) * grid_dy
        goal_lat = ymax - (gy * scale + scale // 2) * grid_dy
        dlat = abs(cy - gy) * scale * grid_dy * 60
        dx_cells = abs(cx - gx)
        dx_cells = min(dx_cells, w - dx_cells)
        dlon = dx_cells * scale * grid_dx * 60 * math.cos(math.radians((lat + goal_lat) / 2))
        return math.sqrt(dlat**2 + dlon**2)
    
    open_set = [(heuristic(sx, sy), (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed:
            continue
        closed.add(current)
        
        if current == (gx, gy):
            # Reconstruct path - return fine grid coordinates (center of each coarse cell)
            path = []
            node = current
            while node is not None:
                cx, cy = node
                # Convert to fine grid coords (center of coarse cell)
                fine_x = cx * scale + scale // 2
                fine_y = cy * scale + scale // 2
                path.append((fine_x, fine_y))
                node = came_from.get(node)
            path.reverse()
            return path
        
        cx, cy = current
        cur_lat = ymax - (cy * scale + scale // 2) * grid_dy
        lat_factor = math.cos(math.radians(cur_lat))
        
        for dx, dy, mult in MOVES_8:
            nx, ny = int(cx + dx), int(cy + dy)
            if not (0 <= ny < h):
                continue
            nx %= w
            if not local_mask[ny, nx]:
                continue
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            
            # Cost in approximate NM
            step_nm = mult * scale * math.sqrt((grid_dx * 60 * lat_factor)**2 + (grid_dy * 60)**2) / 1.414
            tentative_g = g_score[current] + step_nm
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_set, (f, neighbor))
    
    return None


def _draw_thick_line(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, width: int) -> None:
    """Draw a thick line on a 2D array using Bresenham's algorithm with width."""
    h, w = arr.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    while True:
        # Fill square around current point
        for wy in range(max(0, y - width), min(h, y + width + 1)):
            for wx in range(max(0, x - width), min(w, x + width + 1)):
                arr[wy, wx] = 1
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _build_corridor_from_path(
    coarse_path: List[Tuple[int, int]],
    grid_shape: Tuple[int, int],
    land_mask: Optional[object],
    corridor_width: int
) -> Tuple[np.ndarray, int, int]:
    """Build a narrow corridor mask around a coarse path.
    
    Uses simple expansion around each waypoint (faster and tighter than thick lines).
    
    Returns (corridor_mask, x_offset, y_offset) for the bounding box.
    """
    if not coarse_path:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0
    
    # Find bounding box of the path with padding
    xs = [p[0] for p in coarse_path]
    ys = [p[1] for p in coarse_path]
    
    padding = corridor_width + 4  # Extra margin for routing
    x_min = max(0, min(xs) - padding)
    x_max = min(grid_shape[1], max(xs) + padding + 1)
    y_min = max(0, min(ys) - padding)
    y_max = min(grid_shape[0], max(ys) + padding + 1)
    
    # Create corridor mask for bounding box only
    h, w = y_max - y_min, x_max - x_min
    corridor = np.zeros((h, w), dtype=np.uint8)
    
    # Simple expansion around each waypoint
    for px, py in coarse_path:
        lx, ly = px - x_min, py - y_min
        y0 = max(0, ly - corridor_width)
        y1 = min(h, ly + corridor_width + 1)
        x0 = max(0, lx - corridor_width)
        x1 = min(w, lx + corridor_width + 1)
        corridor[y0:y1, x0:x1] = 1
    
    # Subtract land if provided
    if land_mask is not None:
        if hasattr(land_mask, "window"):
            land_region = land_mask.window(y_min, x_min, h, w)
        else:
            land_region = land_mask[y_min:y_max, x_min:x_max]
        corridor = corridor & (land_region == 0)
    
    return corridor, x_min, y_min


def _draw_circle(mask: np.ndarray, cx: int, cy: int, radius: int, h: int, w: int):
    """Draw a filled circle on the mask."""
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)
    
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                mask[y, x] = 1


class CoarseToFineAStar:
    """
    Two-phase A*: 
    1. Coarse search on full grid (no corridor needed)
    2. Fine search in narrow corridor around coarse path
    """
    
    def __init__(
        self,
        grid: GridSpec,
        bathy: "Bathy" | np.ndarray,
        land_mask: Optional["LandMask" | np.ndarray] = None,
        coarse_scale: int = 32,  # Reduced from 64 for better resolution
        corridor_width_nm: float = 10.0,  # Reduced from 25.0 for narrower corridor
        coarse_override: Optional[np.ndarray] = None,
        canals: Optional[object] = None,
    ):
        self.grid = grid
        self.bathy = _ArrayBathy(bathy) if isinstance(bathy, np.ndarray) else bathy
        if isinstance(land_mask, np.ndarray):
            self.land_mask = _ArrayLandMask(land_mask)
        else:
            self.land_mask = land_mask
        self.coarse_scale = coarse_scale
        # Convert corridor width from NM to fine-grid cells using grid.dx
        # (degrees -> NM conversion: 1 degree ≈ 60 NM)
        cell_nm = max(self.grid.dx * 60.0, 1e-6)
        computed_cells = int(round(corridor_width_nm / cell_nm))
        # Ensure a minimum corridor size based on coarse scale
        self.corridor_width_cells = max(computed_cells, coarse_scale * 2)
        
        # Pre-build coarse mask (one-time cost)
        self._coarse_mask = None
        self._coarse_override = coarse_override
        self._canals = canals
    
    @property
    def coarse_mask(self) -> np.ndarray:
        if self._coarse_mask is None:
            self._coarse_mask = _build_global_coarse_mask(
                self.bathy,
                self.land_mask,
                min_draft=1.0,  # Use minimal draft for coarse - refine later
                scale=self.coarse_scale,
                override_mask=self._coarse_override,
            )
        return self._coarse_mask
    
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
        
        # Phase 1: Coarse search on full grid
        coarse_path = _global_coarse_astar(
            self.coarse_mask,
            (start_x, start_y),
            (goal_x, goal_y),
            self.coarse_scale,
            self.grid.dx,
            self.grid.dy,
            self.grid.ymax,
            bathy=self.bathy,
            land=self.land_mask,
            min_draft=min_depth
        )
        
        if coarse_path is None:
            return AStarResult(path=[], explored=0, cost=float('inf'), success=False)
        
        # Phase 2+: Run fine-grained A* segment-by-segment along the coarse path.
        # This avoids building one huge bounding-box corridor that spans the
        # entire route (which would make the fine search very large). Instead
        # we route local segments between successive coarse waypoints and
        # stitch the resulting local paths.
        # Ensure the coarse path is augmented with actual start/end
        aug_path = []
        aug_path.append((start_x, start_y))
        for p in coarse_path:
            # coarse_path entries are (fine_x, fine_y)
            aug_path.append(p)
        aug_path.append((goal_x, goal_y))

        full_lonlats: List[Tuple[float, float]] = []
        total_explored = 0
        total_cost = 0.0

        for i in range(len(aug_path) - 1):
            p0 = aug_path[i]
            p1 = aug_path[i + 1]

            # Build a small corridor around this segment only
            segment_corridor, x_off, y_off = _build_corridor_from_path(
                [p0, p1], (self.grid.height, self.grid.width), None, self.corridor_width_cells
            )

            sx_seg = p0[0] - x_off
            sy_seg = p0[1] - y_off
            gx_seg = p1[0] - x_off
            gy_seg = p1[1] - y_off

            h_seg, w_seg = segment_corridor.shape
            if not (0 <= sy_seg < h_seg and 0 <= sx_seg < w_seg and 0 <= gy_seg < h_seg and 0 <= gx_seg < w_seg):
                return AStarResult(path=[], explored=total_explored, cost=float('inf'), success=False)

            # Ensure start/goal inside corridor
            segment_corridor[sy_seg, sx_seg] = 1
            segment_corridor[gy_seg, gx_seg] = 1

            res = self._fine_astar(
                segment_corridor, (sx_seg, sy_seg), (gx_seg, gy_seg),
                x_off, y_off, context, weights, min_depth, goal_lonlat
            )

            total_explored += res.explored
            total_cost += res.cost if res.success else 0.0

            if not res.success:
                return AStarResult(path=[], explored=total_explored, cost=float('inf'), success=False)

            # Append segment path, avoid duplicating the first point except for first segment
            if not full_lonlats:
                full_lonlats.extend(res.path)
            else:
                full_lonlats.extend(res.path[1:])

        return AStarResult(path=full_lonlats, explored=total_explored, cost=total_cost, success=True)
    
    def _fine_astar(
        self,
        corridor: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        x_off: int,
        y_off: int,
        context: CostContext,
        weights: CostWeights,
        min_depth: float,
        goal_lonlat: Tuple[float, float]
    ) -> AStarResult:
        """Fine-grained A* search in narrow corridor."""
        h, w = corridor.shape
        sx, sy = start
        gx, gy = goal
        
        INF = 1e30
        g_score = np.full((h, w), INF, dtype=np.float32)
        g_score[sy, sx] = 0.0
        
        came_from_x = np.full((h, w), -1, dtype=np.int32)
        came_from_y = np.full((h, w), -1, dtype=np.int32)
        prev_bearing = np.full((h, w), -1.0, dtype=np.float32)
        closed = np.zeros((h, w), dtype=np.uint8)
        
        goal_lon, goal_lat = goal_lonlat
        
        def heuristic(x: int, y: int) -> float:
            lon, lat = self.grid.xy_to_lonlat(x + x_off, y + y_off)
            dlat = abs(lat - goal_lat) * 60
            dlon = abs(shortest_dlon(lon, goal_lon)) * 60 * math.cos(math.radians((lat + goal_lat) / 2))
            return math.sqrt(dlat**2 + dlon**2)
        
        open_set = [(heuristic(sx, sy), (sx, sy))]
        explored = 0
        override_mask = None
        if self._canals:
            from ocean_router.data.canals import canal_mask_window
            override_mask = canal_mask_window(self._canals, self.grid, x_off, y_off, h, w)

        pre = precompute_corridor_arrays(
            corridor,
            x_off,
            y_off,
            context=context,
            min_draft=min_depth,
            weights=weights,
            override_mask=override_mask,
            corridor_path=[(sx + x_off, sy + y_off), (gx + x_off, gy + y_off)],
            grid=self.grid,
        )
        blocked_mask = pre.get("blocked_mask", np.zeros_like(corridor, dtype=bool))
        depth_penalty = pre.get("depth_penalty")
        land_prox_penalty = pre.get("land_prox_penalty")
        tss_in_or_near = pre.get("tss_in_or_near")
        tss_in_lane = pre.get("tss_in_lane")
        tss_correct_near = pre.get("tss_correct_near")
        tss_sepzone = pre.get("tss_sepzone")
        tss_sepboundary = pre.get("tss_sepboundary")
        corridor_bearing = pre.get("corridor_bearing")
        snap_mask = pre.get("snap_corridor_mask")
        corridor_mask = corridor if snap_mask is None or not snap_mask.size else snap_mask
        goal_x_g = x_off + gx
        goal_y_g = y_off + gy
        wrap_x = x_off == 0 and w == self.grid.width
        row_step_nms = _precompute_row_step_nms(self.grid, y_off, h)
        goal_bearing_grid = _precompute_goal_bearing(x_off, y_off, w, h, goal_x_g, goal_y_g)

        def line_is_valid(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
            def is_blocked(x: int, y: int) -> bool:
                ly = y - y_off
                lx = x - x_off
                if not (0 <= ly < h and 0 <= lx < w):
                    return True
                if not corridor_mask[ly, lx]:
                    return True
                if blocked_mask[ly, lx]:
                    return True
                if self.land_mask is not None and self.land_mask.sample(y, x):
                    if override_mask is None or not override_mask[ly, lx]:
                        return True
                return False

            if not _line_of_sight_clear(p0, p1, is_blocked):
                return False
            if context.tss is not None:
                x0, y0 = p0
                x1, y1 = p1
                if context.tss.line_crosses_boundary(y0, x0, y1, x1):
                    return False
                if context.tss_wrong_way_hard and context.tss.line_goes_wrong_way(y0, x0, y1, x1, grid=context.grid):
                    return False
            return True
        
        while open_set:
            _, (cx, cy) = heapq.heappop(open_set)
            
            if closed[cy, cx]:
                continue
            closed[cy, cx] = 1
            explored += 1
            
            if (cx, cy) == (gx, gy):
                # Reconstruct path
                path_indices = []
                x, y = gx, gy
                while x >= 0 and y >= 0:
                    path_indices.append((x + x_off, y + y_off))
                    px, py = came_from_x[y, x], came_from_y[y, x]
                    x, y = px, py
                path_indices.reverse()
                
                # Smooth the path using line-of-sight checks
                smoothed_indices = self._smooth_path(path_indices, context, min_depth)
                
                lonlats = [self.grid.xy_to_lonlat(px, py) for px, py in smoothed_indices]
                return AStarResult(
                    path=lonlats,
                    explored=explored,
                    cost=float(g_score[gy, gx]),
                    success=True
                )
            
            cur_g = g_score[cy, cx]
            cur_bearing = prev_bearing[cy, cx]
            
            gx_cur = x_off + cx
            gy_cur = y_off + cy

            for idx, (dx, dy, _) in enumerate(MOVES_8):
                dx, dy = int(dx), int(dy)
                nx, ny = cx + dx, cy + dy

                if not (0 <= ny < h):
                    continue
                if wrap_x:
                    nx %= w
                elif not (0 <= nx < w):
                    continue
                if closed[ny, nx]:
                    continue
                if not corridor_mask[ny, nx]:
                    continue
                
                gnx = x_off + nx
                gny = y_off + ny
                
                # Check land mask first (faster than blocked)
                if self.land_mask is not None and self.land_mask.sample(gny, gnx):
                    if override_mask is None or not override_mask[ny, nx]:
                        continue

                # Use precomputed blocked mask
                if blocked_mask[ny, nx]:
                    continue
                if context.tss and context.tss_wrong_way_hard:
                    gcx = x_off + cx
                    gcy = y_off + cy
                    if context.tss.line_goes_wrong_way(gcy, gcx, gny, gnx, grid=context.grid):
                        continue
                
                base_cx, base_cy = cx, cy
                base_gx, base_gy = gx_cur, gy_cur
                base_g = cur_g
                base_bearing = cur_bearing
                parent_x = came_from_x[cy, cx]
                parent_y = came_from_y[cy, cx]
                if parent_x >= 0 and parent_y >= 0:
                    parent_gx = x_off + parent_x
                    parent_gy = y_off + parent_y
                    if line_is_valid((parent_gx, parent_gy), (gnx, gny)):
                        base_cx, base_cy = parent_x, parent_y
                        base_gx, base_gy = parent_gx, parent_gy
                        base_g = float(g_score[parent_y, parent_x])
                        base_bearing = prev_bearing[parent_y, parent_x]

                if base_cx == cx and base_cy == cy:
                    move_bearing = float(MOVE_BEARINGS[idx])
                    step_nm = float(row_step_nms[cy, idx])
                else:
                    base_lon, base_lat = self.grid.xy_to_lonlat(base_gx, base_gy)
                    next_lon, next_lat = self.grid.xy_to_lonlat(gnx, gny)
                    move_bearing = bearing_deg(base_lon, base_lat, next_lon, next_lat)
                    step_nm = rhumb_distance_nm(base_lon, base_lat, next_lon, next_lat)
                
                penalty = 0.0
                if depth_penalty is not None:
                    penalty += depth_penalty[ny, nx]
                else:
                    penalty += context.bathy.depth_penalty(gny, gnx, min_depth, weights.near_shore_depth_penalty)
                
                # Add land proximity penalty
                if land_prox_penalty is not None:
                    penalty += land_prox_penalty[ny, nx]
                else:
                    if context.land:
                        penalty += context.land.proximity_penalty(gny, gnx, weights.land_proximity_penalty, max_distance_cells=weights.land_proximity_max_distance_cells)
                
                # Add TSS penalties (only if near a TSS lane)
                if context.tss:
                    gcx = base_gx
                    gcy = base_gy
                    if tss_in_or_near is not None:
                        in_near = bool(tss_in_or_near[ny, nx])
                        prev_in_near = bool(tss_in_or_near[base_cy, base_cx])
                    else:
                        in_near = context.tss.in_or_near_lane(gny, gnx, radius=weights.tss_proximity_check_radius)
                        prev_in_near = context.tss.in_or_near_lane(gcy, gcx, radius=weights.tss_proximity_check_radius)

                    if weights.tss_off_lane_penalty > 0:
                        in_lane = False
                        if tss_in_lane is not None and tss_in_lane.size:
                            in_lane = bool(tss_in_lane[ny, nx])
                        elif context.tss.in_lane(gny, gnx):
                            in_lane = True
                        if (in_near or prev_in_near) and not in_lane:
                            if tss_correct_near is not None and bool(tss_correct_near[ny, nx]):
                                penalty += weights.tss_off_lane_penalty

                    if in_near or prev_in_near:
                        goal_bearing = None
                        if corridor_bearing is not None and corridor_bearing.size:
                            cb = corridor_bearing[ny, nx]
                            if cb >= 0:
                                goal_bearing = float(cb)
                        if goal_bearing is None:
                            goal_bearing = float(goal_bearing_grid[cy, cx])
                        penalty += context.tss.alignment_penalty(
                            gny, gnx, move_bearing,
                            weights.tss_wrong_way_penalty,
                            weights.tss_alignment_weight,
                            prev_y=gcy,
                            prev_x=gcx,
                            goal_bearing=goal_bearing,
                            max_lane_deviation_deg=weights.tss_max_lane_deviation_deg,
                            proximity_check_radius=weights.tss_proximity_check_radius,
                            grid=context.grid,
                        )
                    # Always apply boundary crossing penalties to avoid cutting across separation lines.
                    if (
                        tss_in_lane is not None
                        and tss_in_lane.size
                        and tss_sepzone is not None
                        and tss_sepzone.size
                        and tss_sepboundary is not None
                        and tss_sepboundary.size
                    ):
                        was_in_lane = bool(tss_in_lane[base_cy, base_cx])
                        now_in_lane = bool(tss_in_lane[ny, nx])
                        if was_in_lane and not now_in_lane:
                            penalty += weights.tss_lane_crossing_penalty
                        was_in_sepzone = bool(tss_sepzone[base_cy, base_cx])
                        now_in_sepzone = bool(tss_sepzone[ny, nx])
                        if not was_in_sepzone and now_in_sepzone:
                            penalty += weights.tss_sepzone_crossing_penalty
                        elif was_in_sepzone and not now_in_sepzone:
                            penalty += weights.tss_sepzone_crossing_penalty * 0.25
                        crossed_boundary = was_in_sepzone != now_in_sepzone
                        prev_on_sepboundary = bool(tss_sepboundary[base_cy, base_cx])
                        now_on_sepboundary = bool(tss_sepboundary[ny, nx])
                        if not prev_on_sepboundary and now_on_sepboundary:
                            penalty += weights.tss_sepboundary_crossing_penalty
                            crossed_boundary = True
                        if crossed_boundary:
                            penalty += weights.tss_sepzone_crossing_penalty
                    else:
                        penalty += context.tss.boundary_crossing_penalty(
                            gcy, gcx, gny, gnx,
                            weights.tss_lane_crossing_penalty,
                            weights.tss_sepzone_crossing_penalty,
                            weights.tss_sepboundary_crossing_penalty
                        )
                
                if base_bearing >= 0:
                    angle_diff = abs(((move_bearing - base_bearing + 180) % 360) - 180)
                    penalty += weights.turn_penalty_weight * angle_diff / 180
                
                tentative_g = base_g + step_nm + penalty
                
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    came_from_x[ny, nx] = base_cx
                    came_from_y[ny, nx] = base_cy
                    prev_bearing[ny, nx] = move_bearing
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, (nx, ny)))
        
        return AStarResult(path=[], explored=explored, cost=float('inf'), success=False)
    
    def _smooth_path(
        self,
        path_xy: List[Tuple[int, int]],
        context: CostContext,
        min_depth: float,
        max_skip: int = 100,
        min_land_distance_cells: int = 5
    ) -> List[Tuple[int, int]]:
        """Smooth path using line-of-sight checks (string pulling).
        
        Respects TSS boundaries - won't smooth across lane boundaries or into separation zones.
        Also ensures smoothed path maintains safe distance from land (unless in TSS lanes).
        """
        if len(path_xy) < 3:
            return path_xy
        
        def is_blocked(x: int, y: int) -> bool:
            if not (0 <= y < self.bathy.depth.shape[0] and 0 <= x < self.bathy.depth.shape[1]):
                return True
            if self.land_mask is not None and self.land_mask.sample(y, x):
                return True
            if context.blocked(y, x, min_depth):
                return True
            return False
        
        def line_is_valid(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
            """Check if line between two points is valid (no obstacles, no TSS boundary crossings, safe distance from land)."""
            # First check standard line-of-sight (obstacles)
            if not _line_of_sight_clear(p0, p1, is_blocked):
                return False
            # Check if line passes too close to land (TSS lanes are exempt)
            if context.land is not None:
                if _line_too_close_to_land(p0, p1, context.land.distance_from_land, min_land_distance_cells, context.tss):
                    return False
            # Only check separation zone/boundary crossings (not lane boundaries or wrong-way)
            # A* already ensured the path follows correct TSS directions
            if context.tss is not None:
                x0, y0 = p0
                x1, y1 = p1
                if context.tss_disable_lane_smoothing and context.tss.line_intersects_lane(y0, x0, y1, x1):
                    return False
                if context.tss.line_crosses_boundary(y0, x0, y1, x1):
                    return False
            return True
        
        smoothed = _smooth_path_with_validator(path_xy, line_is_valid, max_skip)
        return smoothed


class FastCorridorAStar:
    """Optimized single-level A* with NumPy arrays instead of dicts."""

    def __init__(self, grid: GridSpec, corridor_mask: np.ndarray, x_off: int, y_off: int, precomputed: Optional[dict] = None):
        self.grid = grid
        self.corridor_mask = corridor_mask
        self.x_off = x_off
        self.y_off = y_off
        self.precomputed = precomputed or {}
        
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
        
        # Convert to corridor-local coordinates
        sx, sy = start_x - self.x_off, start_y - self.y_off
        gx, gy = goal_x - self.x_off, goal_y - self.y_off
        
        corridor_mask = self.precomputed.get("snap_corridor_mask") if hasattr(self, "precomputed") else None
        if corridor_mask is None or not corridor_mask.size:
            corridor_mask = self.corridor_mask
        h, w = corridor_mask.shape
        if not (0 <= sy < h and 0 <= sx < w and corridor_mask[sy, sx]):
            raise ValueError("start outside corridor")
        if not (0 <= gy < h and 0 <= gx < w and corridor_mask[gy, gx]):
            raise ValueError("goal outside corridor")
        
        # Use numpy arrays for g_score and came_from
        INF = 1e30
        g_score = np.full((h, w), INF, dtype=np.float32)
        g_score[sy, sx] = 0.0
        
        came_from_x = np.full((h, w), -1, dtype=np.int32)
        came_from_y = np.full((h, w), -1, dtype=np.int32)
        prev_bearing = np.full((h, w), -1.0, dtype=np.float32)
        
        # Closed set as boolean array
        closed = np.zeros((h, w), dtype=np.uint8)
        
        goal_lon, goal_lat = goal_lonlat
        
        def heuristic(x: int, y: int) -> float:
            px = self.x_off + x
            py = self.y_off + y
            lon, lat = self.grid.xy_to_lonlat(px, py)
            # Fast approximate distance
            dlat = abs(lat - goal_lat) * 60
            dlon = abs(shortest_dlon(lon, goal_lon)) * 60 * math.cos(math.radians((lat + goal_lat) / 2))
            return math.sqrt(dlat**2 + dlon**2)
        
        open_set = [(heuristic(sx, sy), (sx, sy))]
        explored = 0
        corridor_bearing = self.precomputed.get('corridor_bearing') if hasattr(self, 'precomputed') else None
        tss_in_lane = self.precomputed.get('tss_in_lane') if hasattr(self, 'precomputed') else None
        tss_correct_near = self.precomputed.get('tss_correct_near') if hasattr(self, 'precomputed') else None
        tss_in_or_near = self.precomputed.get('tss_in_or_near') if hasattr(self, 'precomputed') else None
        tss_sepzone = self.precomputed.get('tss_sepzone') if hasattr(self, 'precomputed') else None
        tss_sepboundary = self.precomputed.get('tss_sepboundary') if hasattr(self, 'precomputed') else None
        wrap_x = self.x_off == 0 and w == self.grid.width
        row_step_nms = _precompute_row_step_nms(self.grid, self.y_off, h)
        goal_bearing_grid = _precompute_goal_bearing(self.x_off, self.y_off, w, h, self.x_off + gx, self.y_off + gy)
        
        # Timing counters for profiling
        time_heappop = 0.0
        time_neighbor_gen = 0.0
        time_blocked_check = 0.0
        time_cost_calc = 0.0
        time_heappush = 0.0
        sample_interval = 10**9  # effectively disable periodic timing prints
        
        while open_set:
            t0 = time.perf_counter()
            _, (cx, cy) = heapq.heappop(open_set)
            time_heappop += time.perf_counter() - t0
            
            if closed[cy, cx]:
                continue
            closed[cy, cx] = 1
            explored += 1
            
            # Periodic timing report
            if explored % sample_interval == 0:
                total_time = time_heappop + time_neighbor_gen + time_blocked_check + time_cost_calc + time_heappush
                print(f"[A* TIMING @ {explored} nodes] heappop={time_heappop*1000:.1f}ms, neighbor={time_neighbor_gen*1000:.1f}ms, blocked={time_blocked_check*1000:.1f}ms, cost={time_cost_calc*1000:.1f}ms, heappush={time_heappush*1000:.1f}ms, total={total_time*1000:.1f}ms")
            
            if (cx, cy) == (gx, gy):
                # Reconstruct
                path_indices = []
                x, y = gx, gy
                while x >= 0 and y >= 0:
                    path_indices.append((x + self.x_off, y + self.y_off))
                    px, py = came_from_x[y, x], came_from_y[y, x]
                    x, y = px, py
                path_indices.reverse()
                
                # Smooth the path (TSS-aware and land-aware)
                min_land_distance_cells = 5  # Minimum distance from land in grid cells (~5nm)
                
                def is_blocked(x: int, y: int) -> bool:
                    return context.blocked(y, x, min_depth)
                
                def line_is_valid(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
                    if not _line_of_sight_clear(p0, p1, is_blocked):
                        return False
                    # Check if line passes too close to land (TSS lanes are exempt)
                    if context.land is not None:
                        if _line_too_close_to_land(p0, p1, context.land.distance_from_land, min_land_distance_cells, context.tss):
                            return False
                    # Only check separation zone/boundary crossings
                    # A* already ensured the path follows correct TSS directions
                    if context.tss is not None:
                        x0, y0 = p0
                        x1, y1 = p1
                        if context.tss_disable_lane_smoothing and context.tss.line_intersects_lane(y0, x0, y1, x1):
                            return False
                        if context.tss.line_crosses_boundary(y0, x0, y1, x1):
                            return False
                    return True
                
                smoothed = _smooth_path_with_validator(path_indices, line_is_valid)
                
                lonlats = [self.grid.xy_to_lonlat(px, py) for px, py in smoothed]
                return AStarResult(
                    path=lonlats,
                    explored=explored,
                    cost=float(g_score[gy, gx]),
                    success=True
                )
            
            cur_g = g_score[cy, cx]
            cur_bearing = prev_bearing[cy, cx]
            
            gx_cur = self.x_off + cx
            gy_cur = self.y_off + cy

            for idx, (dx, dy, _) in enumerate(MOVES_8):
                dx, dy = int(dx), int(dy)
                nx, ny = cx + dx, cy + dy
                
                t0 = time.perf_counter()
                if not (0 <= ny < h):
                    continue
                if wrap_x:
                    nx %= w
                elif not (0 <= nx < w):
                    continue
                if closed[ny, nx]:
                    continue
                if not corridor_mask[ny, nx]:
                    continue
                time_neighbor_gen += time.perf_counter() - t0
                
                gnx = self.x_off + nx
                gny = self.y_off + ny
                
                t0 = time.perf_counter()
                # Use precomputed blocked mask if available
                blocked_here = False
                bm = self.precomputed.get('blocked_mask') if hasattr(self, 'precomputed') else None
                if bm is not None and bm.size:
                    if bm[ny, nx]:
                        time_blocked_check += time.perf_counter() - t0
                        continue
                else:
                    if context.blocked(gny, gnx, min_depth):
                        time_blocked_check += time.perf_counter() - t0
                        continue
                time_blocked_check += time.perf_counter() - t0

                if context.tss and context.tss_wrong_way_hard:
                    if context.tss.line_goes_wrong_way(gy_cur, gx_cur, gny, gnx, grid=context.grid):
                        continue
                
                t0 = time.perf_counter()
                move_bearing = float(MOVE_BEARINGS[idx])
                step_nm = float(row_step_nms[cy, idx])
                
                penalty = 0.0
                # Use precomputed depth/land penalties if available
                dp = self.precomputed.get('depth_penalty') if hasattr(self, 'precomputed') else None
                if dp is not None and dp.size:
                    penalty += dp[ny, nx]
                else:
                    penalty += context.bathy.depth_penalty(gny, gnx, min_depth, weights.near_shore_depth_penalty)

                lp = self.precomputed.get('land_prox_penalty') if hasattr(self, 'precomputed') else None
                if lp is not None and lp.size:
                    penalty += lp[ny, nx]
                else:
                    if context.land:
                        penalty += context.land.proximity_penalty(gny, gnx, weights.land_proximity_penalty, max_distance_cells=weights.land_proximity_max_distance_cells)
                
                # Add TSS penalties (only if near a TSS lane)
                if context.tss:
                    # Use precomputed flag to skip in_or_near checks when possible
                    in_near = False
                    prev_in_near = False
                    if tss_in_or_near is not None and tss_in_or_near.size:
                        in_near = bool(tss_in_or_near[ny, nx])
                        prev_in_near = bool(tss_in_or_near[cy, cx])
                    else:
                        in_near = context.tss.in_or_near_lane(gny, gnx, radius=weights.tss_proximity_check_radius)
                        prev_in_near = context.tss.in_or_near_lane(gy_cur, gx_cur, radius=weights.tss_proximity_check_radius)

                    if weights.tss_off_lane_penalty > 0:
                        in_lane = False
                        if tss_in_lane is not None and tss_in_lane.size:
                            in_lane = bool(tss_in_lane[ny, nx])
                        elif context.tss.in_lane(gny, gnx):
                            in_lane = True
                        if (in_near or prev_in_near) and not in_lane:
                            if tss_correct_near is not None and bool(tss_correct_near[ny, nx]):
                                penalty += weights.tss_off_lane_penalty

                    if in_near or prev_in_near:
                        goal_bearing = None
                        if corridor_bearing is not None and corridor_bearing.size:
                            cb = corridor_bearing[ny, nx]
                            if cb >= 0:
                                goal_bearing = float(cb)
                        if goal_bearing is None:
                            goal_bearing = float(goal_bearing_grid[cy, cx])
                        penalty += context.tss.alignment_penalty(
                            gny, gnx, move_bearing,
                            weights.tss_wrong_way_penalty,
                            weights.tss_alignment_weight,
                            prev_y=gy_cur,
                            prev_x=gx_cur,
                            goal_bearing=goal_bearing,
                            max_lane_deviation_deg=weights.tss_max_lane_deviation_deg,
                            proximity_check_radius=weights.tss_proximity_check_radius,
                            grid=context.grid,
                        )
                    # Always apply boundary crossing penalties to avoid cutting across separation lines.
                    if (
                        tss_in_lane is not None
                        and tss_in_lane.size
                        and tss_sepzone is not None
                        and tss_sepzone.size
                        and tss_sepboundary is not None
                        and tss_sepboundary.size
                    ):
                        was_in_lane = bool(tss_in_lane[cy, cx])
                        now_in_lane = bool(tss_in_lane[ny, nx])
                        if was_in_lane and not now_in_lane:
                            penalty += weights.tss_lane_crossing_penalty
                        was_in_sepzone = bool(tss_sepzone[cy, cx])
                        now_in_sepzone = bool(tss_sepzone[ny, nx])
                        if not was_in_sepzone and now_in_sepzone:
                            penalty += weights.tss_sepzone_crossing_penalty
                        elif was_in_sepzone and not now_in_sepzone:
                            penalty += weights.tss_sepzone_crossing_penalty * 0.25
                        crossed_boundary = was_in_sepzone != now_in_sepzone
                        prev_on_sepboundary = bool(tss_sepboundary[cy, cx])
                        now_on_sepboundary = bool(tss_sepboundary[ny, nx])
                        if not prev_on_sepboundary and now_on_sepboundary:
                            penalty += weights.tss_sepboundary_crossing_penalty
                            crossed_boundary = True
                        if crossed_boundary:
                            penalty += weights.tss_sepzone_crossing_penalty
                    else:
                        penalty += context.tss.boundary_crossing_penalty(
                            gy_cur, gx_cur, gny, gnx,
                            weights.tss_lane_crossing_penalty,
                            weights.tss_sepzone_crossing_penalty,
                            weights.tss_sepboundary_crossing_penalty
                        )
                
                if cur_bearing >= 0:
                    angle_diff = abs(((move_bearing - cur_bearing + 180) % 360) - 180)
                    penalty += weights.turn_penalty_weight * angle_diff / 180
                
                tentative_g = cur_g + step_nm + penalty
                time_cost_calc += time.perf_counter() - t0
                
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    came_from_x[ny, nx] = cx
                    came_from_y[ny, nx] = cy
                    prev_bearing[ny, nx] = move_bearing
                    f = tentative_g + heuristic(nx, ny) * heuristic_weight
                    t0 = time.perf_counter()
                    heapq.heappush(open_set, (f, (nx, ny)))
                    time_heappush += time.perf_counter() - t0
        
        # Final timing report
        total_time = time_heappop + time_neighbor_gen + time_blocked_check + time_cost_calc + time_heappush
        print(f"[A* FINAL TIMING] explored={explored}, heappop={time_heappop*1000:.1f}ms, neighbor={time_neighbor_gen*1000:.1f}ms, blocked={time_blocked_check*1000:.1f}ms, cost={time_cost_calc*1000:.1f}ms, heappush={time_heappush*1000:.1f}ms, total={total_time*1000:.1f}ms")
        
        return AStarResult(path=[], explored=explored, cost=float('inf'), success=False)
