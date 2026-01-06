"""Polyline simplification helpers."""
from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import math

from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    from ocean_router.data.tss import TSSFields
    from ocean_router.data.land import LandMask
    from ocean_router.core.grid import GridSpec


def perpendicular_distance_nm(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line in nautical miles.
    
    Args:
        point: (lon, lat)
        line_start: (lon, lat)
        line_end: (lon, lat)
    
    Returns:
        Distance in nautical miles
    """
    lon, lat = point
    lon1, lat1 = line_start
    lon2, lat2 = line_end
    
    # Convert to approximate nm coordinates (1 degree lat ≈ 60 nm)
    x = lon * 60 * math.cos(math.radians(lat))
    y = lat * 60
    x1 = lon1 * 60 * math.cos(math.radians(lat1))
    y1 = lat1 * 60
    x2 = lon2 * 60 * math.cos(math.radians(lat2))
    y2 = lat2 * 60
    
    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    
    # If line is a point, return distance to that point
    line_length_sq = dx * dx + dy * dy
    if line_length_sq == 0:
        return math.sqrt((x - x1)**2 + (y - y1)**2)
    
    # Calculate perpendicular distance
    # Project point onto line and find perpendicular distance
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / line_length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)


def simplify_between_tss_boundaries(
    points: List[Tuple[float, float]], 
    grid: 'GridSpec',
    tss_fields: Optional['TSSFields'] = None,
    tolerance_nm: float = 0.5,
    land_mask: Optional['LandMask'] = None,
    min_land_distance_cells: int = 10
) -> List[Tuple[float, float]]:
    """Simplify path by keeping TSS entry/exit points and removing collinear points.
    
    Args:
        points: List of (lon, lat) tuples
        grid: GridSpec for coordinate conversion
        tss_fields: TSSFields to detect boundary crossings
        tolerance_nm: Remove points within this distance from the line between boundaries
        land_mask: Optional land mask to check proximity to land
        min_land_distance_cells: Minimum distance from land to maintain (in grid cells)
        
    Returns:
        Simplified list of (lon, lat) tuples
    """
    if len(points) < 3 or tss_fields is None:
        return points
    
    # Identify TSS entry/exit points (entering or leaving the TSS area entirely)
    # and separation zone/boundary crossings (critical navigation points)
    boundary_indices = [0]  # Always keep start
    
    for i in range(1, len(points) - 1):
        x_prev, y_prev = grid.lonlat_to_xy(points[i-1][0], points[i-1][1])
        x_curr, y_curr = grid.lonlat_to_xy(points[i][0], points[i][1])
        
        # Check if entering or leaving TSS area (lane OR sepzone)
        was_in_tss = tss_fields.in_lane(y_prev, x_prev) or tss_fields.in_sepzone(y_prev, x_prev)
        now_in_tss = tss_fields.in_lane(y_curr, x_curr) or tss_fields.in_sepzone(y_curr, x_curr)
        
        # Check if crossing a separation boundary line
        was_on_sepboundary = tss_fields.in_sepboundary(y_prev, x_prev)
        now_on_sepboundary = tss_fields.in_sepboundary(y_curr, x_curr)
        
        # Keep point if:
        # 1. Entering or exiting TSS area entirely
        # 2. Crossing a separation boundary line
        if was_in_tss != now_in_tss:
            boundary_indices.append(i)
        elif not was_on_sepboundary and now_on_sepboundary:
            boundary_indices.append(i)
    
    boundary_indices.append(len(points) - 1)  # Always keep end
    
    # Now simplify segments between boundary points
    simplified = []
    
    for seg_idx in range(len(boundary_indices) - 1):
        start_idx = boundary_indices[seg_idx]
        end_idx = boundary_indices[seg_idx + 1]
        
        # Always add the segment start
        simplified.append(points[start_idx])
        
        # If segment has intermediate points, apply Ramer-Douglas-Peucker style simplification
        if end_idx - start_idx > 1:
            segment = points[start_idx:end_idx + 1]
            simplified_segment = _simplify_segment(segment, tolerance_nm, grid, land_mask, min_land_distance_cells, tss_fields)
            # Add intermediate points (exclude start and end)
            simplified.extend(simplified_segment[1:-1])
    
    # Add final point
    simplified.append(points[-1])
    
    return simplified


def _simplify_segment(
    segment: List[Tuple[float, float]], 
    tolerance_nm: float,
    grid: Optional['GridSpec'] = None,
    land_mask: Optional['LandMask'] = None,
    min_land_distance_cells: int = 10,
    tss_fields: Optional['TSSFields'] = None
) -> List[Tuple[float, float]]:
    """Simplify a segment using perpendicular distance threshold (iterative version).
    
    Uses stack-based iteration instead of recursion to avoid stack overflow
    on long routes. Removes points that are within tolerance_nm of the line
    between segment endpoints, while preserving points needed to maintain
    safe distance from land (unless in TSS lanes).
    """
    if len(segment) <= 2:
        return segment
    
    # Use a stack-based approach to avoid recursion
    # Each stack item is (start_idx, end_idx) of a segment to process
    # Result is built by tracking which indices to keep
    keep_indices = set([0, len(segment) - 1])  # Always keep endpoints
    
    stack = [(0, len(segment) - 1)]
    
    while stack:
        start_idx, end_idx = stack.pop()
        
        if end_idx - start_idx <= 1:
            continue  # No points between start and end
        
        # Find point with maximum perpendicular distance from line
        max_dist = 0
        max_idx = start_idx + 1
        
        for i in range(start_idx + 1, end_idx):
            dist = perpendicular_distance_nm(segment[i], segment[start_idx], segment[end_idx])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # Check if we need to keep this point
        need_to_keep = False
        
        if max_dist > tolerance_nm:
            # Point is too far from line - must keep it
            need_to_keep = True
        elif land_mask is not None and grid is not None:
            # Check if simplifying would bring line too close to land (TSS lanes exempt)
            if _line_too_close_to_land(segment[start_idx], segment[end_idx], grid, land_mask, min_land_distance_cells, tss_fields):
                need_to_keep = True
        
        if need_to_keep:
            keep_indices.add(max_idx)
            # Process both sub-segments
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))
    
    # Build result from kept indices
    result = [segment[i] for i in sorted(keep_indices)]
    return result


def _line_too_close_to_land(
    start: Tuple[float, float],
    end: Tuple[float, float],
    grid: 'GridSpec',
    land_mask: 'LandMask',
    min_distance_cells: int,
    tss_fields: Optional['TSSFields'] = None
) -> bool:
    """Check if a straight line passes too close to land.
    
    Uses Bresenham's algorithm to check all cells along the line.
    Points inside TSS lanes are exempt (shipping lanes are safe near land).
    """
    x0, y0 = grid.lonlat_to_xy(start[0], start[1])
    x1, y1 = grid.lonlat_to_xy(end[0], end[1])
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        # Check bounds
        if 0 <= y < land_mask.base.shape[0] and 0 <= x < land_mask.base.shape[1]:
            dist_from_land = land_mask.distance_from_land[y, x]
            if dist_from_land < min_distance_cells:
                # Allow if inside a TSS lane (shipping lanes are safe near land)
                if tss_fields is not None and tss_fields.in_lane(y, x):
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


def douglas_peucker(points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    simplified = LineString(points).simplify(tolerance, preserve_topology=True)
    return list(simplified.coords)


def simplify_path(points: List[Tuple[float, float]], tolerance_nm: float = 1.0) -> List[Tuple[float, float]]:
    """Simplify a path using Douglas-Peucker algorithm.
    
    Args:
        points: List of (lon, lat) tuples
        tolerance_nm: Tolerance in nautical miles
        
    Returns:
        Simplified list of (lon, lat) tuples
    """
    if len(points) < 3:
        return points
    
    # Convert nm tolerance to degrees (rough approximation)
    tolerance_deg = tolerance_nm / 60.0
    
    return douglas_peucker(points, tolerance_deg)


def smooth_path_los(
    path_xy: List[Tuple[int, int]],
    is_blocked: callable,
    max_skip: int = 50,
    tss_fields: Optional['TSSFields'] = None
) -> List[Tuple[int, int]]:
    """Smooth a grid path using line-of-sight checks (string pulling).
    
    This removes unnecessary waypoints by checking if we can go directly
    from point A to point C without hitting obstacles, skipping point B.
    
    Args:
        path_xy: List of (x, y) grid coordinates
        is_blocked: Function that takes (x, y) and returns True if blocked
        max_skip: Maximum number of points to try skipping at once
        tss_fields: Optional TSSFields for boundary crossing checks
        
    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path_xy) < 3:
        return path_xy
    
    def line_is_valid(p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
        """Check if line is valid (no obstacles AND no TSS boundary crossings)."""
        if not _line_of_sight(p0, p1, is_blocked):
            return False
        if tss_fields is not None:
            x0, y0 = p0
            x1, y1 = p1
            if tss_fields.line_crosses_boundary(y0, x0, y1, x1):
                return False
        return True
    
    smoothed = [path_xy[0]]
    i = 0
    
    while i < len(path_xy) - 1:
        # Try to skip as many points as possible
        best_j = i + 1
        
        for j in range(min(i + max_skip, len(path_xy) - 1), i + 1, -1):
            if line_is_valid(path_xy[i], path_xy[j]):
                best_j = j
                break
        
        smoothed.append(path_xy[best_j])
        i = best_j
    
    return smoothed


def _line_of_sight(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    is_blocked: callable
) -> bool:
    """Check if there's a clear line of sight between two grid points.
    
    Uses Bresenham's line algorithm to check all cells along the line.
    """
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


def _is_clear_straight_path(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    grid: 'GridSpec',
    land_mask: 'LandMask',
    tss_fields: Optional['TSSFields'] = None,
    min_land_distance_cells: int = 3,
    check_tss_direction: bool = True
) -> bool:
    """Check if a straight line between two lon/lat points is clear.
    
    A path is clear if it:
    - Doesn't cross land
    - Maintains minimum distance from land (unless inside TSS lanes)
    - Doesn't go against TSS traffic direction (if check_tss_direction=True)
    
    Args:
        p0: Start point (lon, lat)
        p1: End point (lon, lat)
        grid: GridSpec for coordinate conversion
        land_mask: Land mask with distance transform
        tss_fields: Optional TSS fields for lane and direction checking
        min_land_distance_cells: Minimum distance from land in grid cells
        check_tss_direction: If True, reject paths that go wrong-way in TSS lanes
        
    Returns:
        True if the straight line is clear
    """
    from ocean_router.core.geodesy import bearing_deg, angle_diff_deg
    
    x0, y0 = grid.lonlat_to_xy(p0[0], p0[1])
    x1, y1 = grid.lonlat_to_xy(p1[0], p1[1])
    
    # Calculate the bearing of this simplified segment
    move_bearing = bearing_deg(p0[1], p0[0], p1[1], p1[0])
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    h, w = land_mask.base.shape
    
    while True:
        # Check bounds
        if 0 <= y < h and 0 <= x < w:
            # Check if on land (blocked)
            if land_mask.base[y, x]:
                return False
            
            # Check distance from land (unless in TSS lane)
            dist_from_land = land_mask.distance_from_land[y, x]
            if dist_from_land < min_land_distance_cells:
                # Allow if in TSS lane
                if tss_fields is not None and tss_fields.in_lane(y, x):
                    pass  # OK - in shipping lane
                else:
                    return False  # Too close to land
            
            # Check TSS direction - reject if going wrong way
            if check_tss_direction and tss_fields is not None:
                if tss_fields.in_lane(y, x):
                    preferred_dir = float(tss_fields.direction_field[y, x])
                    if preferred_dir >= 0:  # Valid direction
                        angle = angle_diff_deg(move_bearing, preferred_dir)
                        if angle > 90:
                            # Going wrong way in TSS lane - not a valid simplification
                            return False
        else:
            # Out of bounds - consider blocked
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


def _check_tss_lane_status(
    path: List[Tuple[float, float]],
    grid: 'GridSpec',
    tss_fields: 'TSSFields'
) -> List[bool]:
    """Check which path points are inside TSS lanes.
    
    Returns:
        List of booleans, True if point is in a TSS lane
    """
    result = []
    for lon, lat in path:
        x, y = grid.lonlat_to_xy(lon, lat)
        if 0 <= y < tss_fields.lane_mask.shape[0] and 0 <= x < tss_fields.lane_mask.shape[1]:
            result.append(tss_fields.in_lane(y, x))
        else:
            result.append(False)
    return result


def tss_aware_simplify(
    path: List[Tuple[float, float]],
    grid: 'GridSpec',
    tss_fields: Optional['TSSFields'],
    land_mask: Optional['LandMask'],
    preserve_points: Optional[List[Tuple[float, float]]] = None,
    max_simplify_nm: float = 50.0,
    min_land_distance_cells: int = 3
) -> Tuple[List[Tuple[float, float]], List[bool]]:
    """Perform TSS-aware path simplification.
    
    This is more sophisticated than simple Douglas-Peucker:
    1. Collapse TSS-to-TSS spans if a straight water path exists
    2. Try to skip intermediate TSS runs (A -> B -> C => go A -> C if clear)
    3. In non-TSS stretches, greedily remove intermediates when a straight clear path exists
    
    Args:
        path: List of (lon, lat) waypoints
        grid: GridSpec for coordinate conversion
        tss_fields: TSS fields for lane detection
        land_mask: Land mask for clearance checking
        preserve_points: Points that must be kept
        max_simplify_nm: Maximum distance to simplify over (prevents skipping islands)
        min_land_distance_cells: Minimum distance from land in grid cells
        
    Returns:
        (simplified_path, in_tss_lane) - simplified path and TSS status for each point
    """
    if not path or len(path) < 3:
        if tss_fields:
            return path, _check_tss_lane_status(path, grid, tss_fields)
        return path, [False] * len(path)
    
    # Convert preserve_points to a set for fast lookup
    preserve_set = set(preserve_points) if preserve_points else set()
    
    # If we don't have land mask or TSS, just return
    if land_mask is None:
        if tss_fields:
            return path, _check_tss_lane_status(path, grid, tss_fields)
        return path, [False] * len(path)
    
    # Convert max_simplify_nm to grid cells
    max_simplify_cells = max_simplify_nm / (grid.dx * 60)  # dx in degrees, 60nm per degree
    
    # Compute TSS flags on the original path
    if tss_fields:
        in_tss_lane = _check_tss_lane_status(path, grid, tss_fields)
    else:
        in_tss_lane = [False] * len(path)
    
    complete_path = list(path)
    
    # --- PHASE 1: Simplify TSS-to-TSS spans ---
    if any(in_tss_lane):
        n = len(complete_path)
        keep_wp = [True] * n
        
        # 1a) Pairwise TSS -> non-TSS -> TSS collapse
        tss_indices = [i for i, f in enumerate(in_tss_lane) if f]
        for i in range(len(tss_indices) - 1):
            s = tss_indices[i]
            e = tss_indices[i + 1]
            if e - s <= 1:
                continue
            # Only consider spans that leave and re-enter TSS
            if not any(not in_tss_lane[k] for k in range(s + 1, e)):
                continue
            # Check if any preserved points are in this span
            has_preserved = any(complete_path[k] in preserve_set for k in range(s + 1, e))
            if has_preserved:
                continue
            # Distance guard
            x0, y0 = grid.lonlat_to_xy(complete_path[s][0], complete_path[s][1])
            x1, y1 = grid.lonlat_to_xy(complete_path[e][0], complete_path[e][1])
            cell_dist = math.hypot(x1 - x0, y1 - y0)
            if cell_dist > max_simplify_cells:
                continue
            # Check if straight path is clear
            if _is_clear_straight_path(complete_path[s], complete_path[e], grid, land_mask, tss_fields, min_land_distance_cells):
                for k in range(s + 1, e):
                    if complete_path[k] not in preserve_set:
                        keep_wp[k] = False
        
        # 1b) Skip middle TSS runs: A -> B -> C => A -> C if clear
        # Build contiguous TSS runs
        tss_runs = []
        idx = 0
        while idx < n:
            if in_tss_lane[idx]:
                run_start = idx
                while idx + 1 < n and in_tss_lane[idx + 1]:
                    idx += 1
                run_end = idx
                tss_runs.append((run_start, run_end))
            idx += 1
        
        # For triples of runs (A, B, C), try going directly from end(A) to start(C)
        for r in range(len(tss_runs) - 2):
            a_start, a_end = tss_runs[r]
            b_start, b_end = tss_runs[r + 1]
            c_start, c_end = tss_runs[r + 2]
            
            s = a_end
            e = c_start
            
            if e - s <= 1:
                continue
            
            has_preserved = any(complete_path[k] in preserve_set for k in range(s + 1, e))
            if has_preserved:
                continue
            
            x0, y0 = grid.lonlat_to_xy(complete_path[s][0], complete_path[s][1])
            x1, y1 = grid.lonlat_to_xy(complete_path[e][0], complete_path[e][1])
            cell_dist = math.hypot(x1 - x0, y1 - y0)
            if cell_dist > max_simplify_cells:
                continue
            
            if _is_clear_straight_path(complete_path[s], complete_path[e], grid, land_mask, tss_fields, min_land_distance_cells):
                for k in range(s + 1, e):
                    if complete_path[k] not in preserve_set:
                        keep_wp[k] = False
        
        # Apply TSS simplification
        if not all(keep_wp):
            complete_path = [p for p, keep in zip(complete_path, keep_wp) if keep]
            in_tss_lane = [f for f, keep in zip(in_tss_lane, keep_wp) if keep]
    
    # --- PHASE 2: Simplify non-TSS stretches ---
    if complete_path and land_mask:
        n = len(complete_path)
        keep_wp2 = [True] * n
        
        idx = 0
        while idx < n:
            if in_tss_lane[idx]:
                idx += 1
                continue
            
            # Start of a non-TSS run
            run_start = max(0, idx - 1)
            while idx < n and not in_tss_lane[idx]:
                idx += 1
            run_end = min(n - 1, idx)
            
            if run_end - run_start >= 2:
                # Greedy simplification from end to start
                i = run_end
                while i > run_start:
                    if not keep_wp2[i]:
                        i -= 1
                        continue
                    
                    best_j = None
                    for j in range(run_start, i):
                        has_preserved = any(complete_path[k] in preserve_set for k in range(j + 1, i))
                        if has_preserved:
                            continue
                        
                        x0, y0 = grid.lonlat_to_xy(complete_path[j][0], complete_path[j][1])
                        x1, y1 = grid.lonlat_to_xy(complete_path[i][0], complete_path[i][1])
                        cell_dist = math.hypot(x1 - x0, y1 - y0)
                        
                        if cell_dist <= max_simplify_cells and _is_clear_straight_path(
                            complete_path[j], complete_path[i], grid, land_mask, tss_fields, min_land_distance_cells
                        ):
                            best_j = j
                            break
                    
                    if best_j is not None:
                        for k in range(best_j + 1, i):
                            if complete_path[k] not in preserve_set:
                                keep_wp2[k] = False
                        i = best_j
                    else:
                        i -= 1
        
        if not all(keep_wp2):
            complete_path = [p for p, keep in zip(complete_path, keep_wp2) if keep]
            in_tss_lane = [f for f, keep in zip(in_tss_lane, keep_wp2) if keep]
    
    return complete_path, in_tss_lane


def smooth_path_spline(
    points: List[Tuple[float, float]],
    num_points: int = 100
) -> List[Tuple[float, float]]:
    """Smooth a path using cubic spline interpolation.
    
    Args:
        points: List of (lon, lat) tuples
        num_points: Number of points in output path
        
    Returns:
        Smoothed path with interpolated points
    """
    if len(points) < 4:
        return points
    
    try:
        from scipy.interpolate import splprep, splev
        
        points_arr = np.array(points)
        
        # Fit spline
        tck, u = splprep([points_arr[:, 0], points_arr[:, 1]], s=0, k=min(3, len(points) - 1))
        
        # Evaluate at evenly spaced points
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        return list(zip(x_new, y_new))
    except ImportError:
        return points
