"""Polyline simplification helpers."""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from shapely.geometry import LineString


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
    max_skip: int = 50
) -> List[Tuple[int, int]]:
    """Smooth a grid path using line-of-sight checks (string pulling).
    
    This removes unnecessary waypoints by checking if we can go directly
    from point A to point C without hitting obstacles, skipping point B.
    
    Args:
        path_xy: List of (x, y) grid coordinates
        is_blocked: Function that takes (x, y) and returns True if blocked
        max_skip: Maximum number of points to try skipping at once
        
    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path_xy) < 3:
        return path_xy
    
    smoothed = [path_xy[0]]
    i = 0
    
    while i < len(path_xy) - 1:
        # Try to skip as many points as possible
        best_j = i + 1
        
        for j in range(min(i + max_skip, len(path_xy) - 1), i + 1, -1):
            if _line_of_sight(path_xy[i], path_xy[j], is_blocked):
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
