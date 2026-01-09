"""Lightweight geodesy helpers."""
from __future__ import annotations

import math
from typing import Tuple


EARTH_RADIUS_M = 6371008.8
NM_PER_METER = 1 / 1852


def haversine_nm(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Legacy API: return a distance in nautical miles.

    NOTE: Great-circle calculations are disabled for this project.
    This function now delegates to `rhumb_distance_nm` to provide a
    consistent, non-great-circle distance measure (rhumb/mercator-based).
    """
    return rhumb_distance_nm(lon1, lat1, lon2, lat2)


def shortest_dlon(lon1: float, lon2: float) -> float:
    """Return the shortest longitudinal delta from lon1 to lon2 in degrees."""
    return (lon2 - lon1 + 180) % 360 - 180


def unwrap_lon(lon: float, ref_lon: float) -> float:
    """Shift lon to be within 180 degrees of ref_lon."""
    return ref_lon + shortest_dlon(ref_lon, lon)


def rhumb_distance_nm(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return rhumb line distance in nautical miles."""
    # Approximate rhumb distance
    dlat = lat2 - lat1
    dlon = shortest_dlon(lon1, lon2)
    lat_avg = (lat1 + lat2) / 2
    # Adjust longitude for convergence
    dlon_adjusted = dlon * math.cos(math.radians(lat_avg))
    distance = math.sqrt(dlat**2 + dlon_adjusted**2) * 60  # 1 degree ≈ 60 NM
    return distance


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return initial bearing from point 1 to point 2 in degrees 0-360."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = math.radians(shortest_dlon(lon1, lon2))
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    brng = (math.degrees(math.atan2(x, y)) + 360) % 360
    return brng


def wrap_lon(lon: float) -> float:
    if lon > 180:
        return lon - 360
    if lon < -180:
        return lon + 360
    return lon


def clamp_lat(lat: float) -> float:
    return max(min(lat, 90.0), -90.0)


def mercator_to_latlon(x: float, y: float) -> Tuple[float, float]:
    """Convert Mercator coordinates to lat/lon."""
    lat = 2 * math.degrees(math.atan(math.exp(y / EARTH_RADIUS_M))) - 90
    lon = math.degrees(x / EARTH_RADIUS_M)
    return lon, lat


def latlon_to_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """Convert lat/lon to Mercator coordinates."""
    x = EARTH_RADIUS_M * math.radians(lon)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def rhumb_interpolate(lon1: float, lat1: float, lon2: float, lat2: float, fraction: float) -> Tuple[float, float]:
    """Interpolate a point along the rhumb line from (lon1, lat1) to (lon2, lat2) at given fraction."""
    lon2 = unwrap_lon(lon2, lon1)
    x1, y1 = latlon_to_mercator(lon1, lat1)
    x2, y2 = latlon_to_mercator(lon2, lat2)
    
    x = x1 + fraction * (x2 - x1)
    y = y1 + fraction * (y2 - y1)
    
    lon, lat = mercator_to_latlon(x, y)
    return wrap_lon(lon), lat


def move_cost_nm(dx_deg: float, dy_deg: float, lat: float) -> float:
    """Approximate move cost in NM for a grid step at a given latitude."""
    lon_scale = math.cos(math.radians(lat))
    dlon_nm = abs(dx_deg) * 60 * lon_scale
    dlat_nm = abs(dy_deg) * 60
    if dx_deg != 0 and dy_deg != 0:
        return math.hypot(dlon_nm, dlat_nm)
    return dlon_nm + dlat_nm


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def interpolate_lonlat(lonlat_a: Tuple[float, float], lonlat_b: Tuple[float, float], weight: float) -> Tuple[float, float]:
    dlon = shortest_dlon(lonlat_a[0], lonlat_b[0])
    lon = lonlat_a[0] + dlon * weight
    lat = lonlat_a[1] + (lonlat_b[1] - lonlat_a[1]) * weight
    return wrap_lon(lon), lat
