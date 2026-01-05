"""Lightweight geodesy helpers."""
from __future__ import annotations

import math
from typing import Tuple


EARTH_RADIUS_M = 6371008.8
NM_PER_METER = 1 / 1852


def haversine_nm(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in nautical miles."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c * NM_PER_METER


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return initial bearing from point 1 to point 2 in degrees 0-360."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
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
    lon = lonlat_a[0] + (lonlat_b[0] - lonlat_a[0]) * weight
    lat = lonlat_a[1] + (lonlat_b[1] - lonlat_a[1]) * weight
    return lon, lat
