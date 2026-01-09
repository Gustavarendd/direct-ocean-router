"""Shapely STRtree index for TSS lane segments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json

from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.strtree import STRtree

from ocean_router.core.geodesy import angle_diff_deg, bearing_deg, rhumb_distance_nm


TSS_LANE_TYPES = {"separation_lane"}
INLAND_WATERWAYS = {"canal", "river"}


@dataclass(frozen=True)
class SegmentMeta:
    flow_bearing_deg: float
    seg_bearing_deg_fwd: float
    length_nm: float
    allowed_dir: int
    lane_feature_id: str | int | None


@dataclass(frozen=True)
class SegmentHit:
    seg_idx: int
    dist_nm: float
    nearest_lon: float
    nearest_lat: float
    flow_bearing_deg: float
    allowed_dir: int


class TSSSegmentIndex:
    def __init__(self, geoms: list[LineString], meta: list[SegmentMeta]) -> None:
        self._geoms = geoms
        self._meta = meta
        self._tree = STRtree(geoms) if geoms else None
        self._geom_id_map = {id(geom): idx for idx, geom in enumerate(geoms)}
        self._cache: dict[tuple[float, float], SegmentHit | None] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @classmethod
    def from_geojson(cls, path: Path) -> "TSSSegmentIndex":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        features = list(data.get("features", []))
        geoms: list[LineString] = []
        meta: list[SegmentMeta] = []
        for feat in features:
            seamark_type = _get_seamark_type(feat)
            if seamark_type not in TSS_LANE_TYPES:
                continue
            if _feature_is_inland(feat, INLAND_WATERWAYS):
                continue
            flow = _get_flow_bearing(feat)
            if flow is None:
                continue
            is_forward = _get_flow_forward(feat)
            lane_id = _get_lane_id(feat)
            geom = shape(feat.get("geometry"))
            for line in _iter_lines(geom):
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                if not is_forward:
                    coords = list(reversed(coords))
                for (lon0, lat0), (lon1, lat1) in _pairwise(coords):
                    length_nm = rhumb_distance_nm(lon0, lat0, lon1, lat1)
                    if length_nm <= 0:
                        continue
                    seg_bearing = bearing_deg(lon0, lat0, lon1, lat1)
                    align_fwd = angle_diff_deg(seg_bearing, flow)
                    align_rev = angle_diff_deg((seg_bearing + 180.0) % 360.0, flow)
                    allowed_dir = 0
                    if align_fwd <= 60:
                        allowed_dir = 1
                    elif align_rev <= 60:
                        allowed_dir = -1
                    if allowed_dir == 0:
                        continue
                    geoms.append(LineString([(lon0, lat0), (lon1, lat1)]))
                    meta.append(
                        SegmentMeta(
                            flow_bearing_deg=float(flow),
                            seg_bearing_deg_fwd=float(seg_bearing),
                            length_nm=float(length_nm),
                            allowed_dir=int(allowed_dir),
                            lane_feature_id=lane_id,
                        )
                    )
        return cls(geoms=geoms, meta=meta)

    def nearest_segment(self, lon: float, lat: float, max_dist_nm: float) -> SegmentHit | None:
        if self._tree is None:
            return None
        key = (round(lon, 4), round(lat, 4))
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        point = Point(lon, lat)
        hits = self._tree.nearest(point)
        if hits is None:
            self._cache[key] = None
            return None
        geom = hits
        idx = self._geom_id_map.get(id(geom))
        if idx is None:
            self._cache[key] = None
            return None
        proj = geom.project(point)
        proj_point = geom.interpolate(proj)
        proj_lon, proj_lat = proj_point.x, proj_point.y
        dist_nm = rhumb_distance_nm(lon, lat, proj_lon, proj_lat)
        if dist_nm > max_dist_nm:
            self._cache[key] = None
            return None
        seg_meta = self._meta[idx]
        hit = SegmentHit(
            seg_idx=idx,
            dist_nm=dist_nm,
            nearest_lon=proj_lon,
            nearest_lat=proj_lat,
            flow_bearing_deg=seg_meta.flow_bearing_deg,
            allowed_dir=seg_meta.allowed_dir,
        )
        self._cache[key] = hit
        return hit

    def penalty_for_move(self, lon0: float, lat0: float, lon1: float, lat1: float) -> float:
        mid_lon = (lon0 + lon1) / 2.0
        mid_lat = (lat0 + lat1) / 2.0
        hit = self.nearest_segment(mid_lon, mid_lat, max_dist_nm=2.0)
        if hit is None:
            return 0.0
        move_bearing = bearing_deg(lon0, lat0, lon1, lat1)
        align = angle_diff_deg(move_bearing, hit.flow_bearing_deg)
        if hit.allowed_dir == -1:
            align = angle_diff_deg((move_bearing + 180.0) % 360.0, hit.flow_bearing_deg)
        if align <= 20:
            return -0.5
        if align >= 90:
            return 5.0
        return (align - 20.0) / 70.0

    def cache_info(self) -> dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses, "size": len(self._cache)}


def _get_seamark_type(feature: dict) -> Optional[str]:
    props = feature.get("properties", {})
    parsed = props.get("parsed_other_tags", {})
    if parsed and "seamark:type" in parsed:
        return parsed["seamark:type"]
    other_tags = props.get("other_tags", "")
    if "seamark:type" in other_tags:
        import re

        match = re.search(r'"seamark:type"=>"([^"]+)"', other_tags)
        if match:
            return match.group(1)
    return props.get("seamark:type")


def _get_flow_bearing(feature: dict) -> Optional[float]:
    props = feature.get("properties", {})
    bearing = props.get("tss_flow_bearing_deg")
    if bearing is None:
        return None
    return float(bearing)


def _get_flow_forward(feature: dict) -> bool:
    props = feature.get("properties", {})
    raw = props.get("tss_flow_is_forward_along_geometry", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in {"true", "1", "yes", "y"}
    return bool(raw)


def _get_lane_id(feature: dict) -> Optional[str]:
    props = feature.get("properties", {})
    for key in ("lane_id", "osm_id", "id", "seamark:id"):
        if key in props:
            return str(props[key])
    return None


def _feature_is_inland(feature: dict, skip_waterways: Iterable[str]) -> bool:
    props = feature.get("properties", {})
    waterway = props.get("waterway")
    if waterway is None:
        parsed = props.get("parsed_other_tags", {})
        if parsed and "waterway" in parsed:
            waterway = parsed.get("waterway")
    if waterway is None:
        other_tags = props.get("other_tags", "")
        if "waterway" in other_tags:
            import re

            match = re.search(r'"waterway"=>"([^"]+)"', other_tags)
            if match:
                waterway = match.group(1)
    if waterway is None:
        return False
    return str(waterway).lower() in {w.lower() for w in skip_waterways}


def _iter_lines(geom: object) -> Iterable[LineString]:
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield line


def _pairwise(coords: list[tuple[float, float]]) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    for idx in range(len(coords) - 1):
        yield coords[idx], coords[idx + 1]
