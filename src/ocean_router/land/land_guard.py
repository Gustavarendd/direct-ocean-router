"""Hybrid land-avoidance guard: vector validation with local refinement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, Tuple
import hashlib
import json
import math
import pickle

import numpy as np
import rasterio
from rasterio import features
import fiona
from shapely import from_wkb, to_wkb
from shapely.geometry import LineString, Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.prepared import PreparedGeometry, prep
from shapely.strtree import STRtree
from pyproj import CRS, Geod, Transformer

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import MemMapLoader, save_memmap


RoutePoints = list[Tuple[float, float]]
RouteResult = RoutePoints | tuple[RoutePoints, dict]


class RouteProvider(Protocol):
    def route(self, start: Tuple[float, float], end: Tuple[float, float]) -> RouteResult:
        ...


LocalRouterFactory = Callable[[GridSpec, np.ndarray], RouteProvider]

_GEOD = Geod(ellps="WGS84")


@dataclass(slots=True)
class LandIndex:
    geoms: list[BaseGeometry]
    prepared: list[PreparedGeometry]
    tree: STRtree
    bounds: np.ndarray
    crs: str
    geom_id_map: dict


@dataclass(slots=True)
class ValidationReport:
    is_ok: bool
    intersections: list[Tuple[int, int]]
    offending_bbox: Tuple[float, float, float, float] | None
    segment_bboxes: dict[int, Tuple[float, float, float, float]]


@dataclass(slots=True)
class OffendingSegment:
    index: int
    bbox: Tuple[float, float, float, float]


@dataclass(slots=True)
class LandGuardParams:
    corridor_buffer_nm: float = 25.0
    refine_resolution_nm: float = 0.25
    max_refinements: int = 5
    buffer_growth_factor: float = 0.5
    tolerance_m: float = 0.0
    local_cache_dir: Path = Path("data/processed/land/local_cache")
    local_router_factory: LocalRouterFactory | None = None


def build_land_index(polygons_path: Path, cache_path: Path | None = None) -> LandIndex:
    """Build a prepared STRtree index from a land polygon dataset.

    Uses a cache file (WKB + bounds) to avoid re-parsing shapefiles on repeat runs.
    """
    cache_path = cache_path or polygons_path.with_suffix(".land_index.pkl")
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        geoms = [from_wkb(wkb) for wkb in payload["wkb"]]
        bounds = np.array(payload["bounds"], dtype=float)
        crs = payload.get("crs", "EPSG:4326")
    else:
        geoms = []
        bounds_list: list[Tuple[float, float, float, float]] = []
        with fiona.open(polygons_path) as src:
            crs = src.crs_wkt or "EPSG:4326"
            for feat in src:
                geom = shape(feat["geometry"])
                if geom.is_empty:
                    continue
                geoms.append(geom)
                bounds_list.append(geom.bounds)
        bounds = np.array(bounds_list, dtype=float)
        payload = {
            "crs": crs,
            "wkb": [to_wkb(geom) for geom in geoms],
            "bounds": bounds.tolist(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)

    tree = STRtree(geoms)
    prepared = [prep(geom) for geom in geoms]
    geom_id_map = {id(g): i for i, g in enumerate(geoms)}
    return LandIndex(geoms=geoms, prepared=prepared, tree=tree, bounds=bounds, crs=crs, geom_id_map=geom_id_map)


def validate_route_no_land(
    route_points: Sequence[Tuple[float, float]],
    land_index: LandIndex,
    tolerance_m: float = 0.0,
) -> ValidationReport:
    """Validate that a route polyline does not intersect land polygons."""
    if len(route_points) < 2:
        return ValidationReport(is_ok=True, intersections=[], offending_bbox=None, segment_bboxes={})

    intersections: list[Tuple[int, int]] = []
    offending_bbox: Tuple[float, float, float, float] | None = None
    segment_bboxes: dict[int, Tuple[float, float, float, float]] = {}

    for idx in range(len(route_points) - 1):
        segment = LineString([route_points[idx], route_points[idx + 1]])
        check_geom = _buffer_segment_m(segment, tolerance_m) if tolerance_m > 0 else segment
        # Query tree for candidate geometries and map to indices
        candidate_geoms = land_index.tree.query(check_geom, predicate="intersects")
        if len(candidate_geoms) == 0:
            continue
        for geom in candidate_geoms:
            geom_idx = land_index.geom_id_map.get(id(geom))
            if geom_idx is None:
                continue
            if land_index.prepared[int(geom_idx)].intersects(check_geom):
                intersections.append((idx, int(geom_idx)))
                seg_bbox = segment.bounds
                offending_bbox = _merge_bbox(offending_bbox, seg_bbox)
                segment_bboxes[idx] = seg_bbox

    return ValidationReport(
        is_ok=len(intersections) == 0,
        intersections=intersections,
        offending_bbox=offending_bbox,
        segment_bboxes=segment_bboxes,
    )


def find_offending_segments(report: ValidationReport) -> list[OffendingSegment]:
    """Extract offending segment indices and bboxes from a validation report."""
    if report.is_ok:
        return []
    unique_segments = sorted({seg_idx for seg_idx, _ in report.intersections})
    offending: list[OffendingSegment] = []
    for seg_idx in unique_segments:
        bbox = report.segment_bboxes.get(seg_idx)
        if bbox is None:
            continue
        offending.append(OffendingSegment(index=seg_idx, bbox=bbox))
    return offending


def refine_route_locally(
    route_points: RoutePoints,
    offending: Sequence[OffendingSegment],
    router_factory: LocalRouterFactory,
    land_index: LandIndex,
    params: LandGuardParams,
    buffer_nm: float,
) -> RoutePoints | None:
    """Re-route locally around offending segments using a higher-resolution mask."""
    if not offending:
        return None

    segment_indices = [segment.index for segment in offending]
    start_anchor_idx, end_anchor_idx = _choose_anchor_indices(
        route_points,
        min(segment_indices),
        max(segment_indices),
        land_index,
        params.tolerance_m,
    )
    if start_anchor_idx >= end_anchor_idx:
        return None

    corridor_bbox = _segments_bbox(route_points, segment_indices)
    corridor_bbox = _expand_bbox_nm(corridor_bbox, buffer_nm)
    corridor_bbox = _expand_bbox_to_points(
        corridor_bbox,
        route_points[start_anchor_idx],
        route_points[end_anchor_idx],
    )

    grid = _build_local_grid(corridor_bbox, params.refine_resolution_nm)
    land_mask = _local_land_mask(grid, land_index, params)

    local_router = router_factory(grid, land_mask)
    local_route_result = local_router.route(route_points[start_anchor_idx], route_points[end_anchor_idx])
    local_route = _route_points_from_result(local_route_result)
    if len(local_route) < 2:
        return None

    stitched = route_points[: start_anchor_idx + 1]
    stitched.extend(local_route[1:-1])
    stitched.extend(route_points[end_anchor_idx:])
    return stitched


def route_with_land_guard(
    start: Tuple[float, float],
    end: Tuple[float, float],
    global_router: RouteProvider,
    land_index: LandIndex,
    params: LandGuardParams,
) -> RoutePoints:
    """Route globally, validate against land vectors, and refine locally as needed."""
    if params.local_router_factory is None:
        raise ValueError("LandGuardParams.local_router_factory is required for refinements")

    global_route_result = global_router.route(start, end)
    route_points = _route_points_from_result(global_route_result)
    if len(route_points) < 2:
        return route_points

    for attempt in range(params.max_refinements):
        report = validate_route_no_land(route_points, land_index, tolerance_m=params.tolerance_m)
        if report.is_ok:
            return route_points

        offending = find_offending_segments(report)
        if not offending:
            break

        buffer_nm = params.corridor_buffer_nm * (1.0 + attempt * params.buffer_growth_factor)
        refined = refine_route_locally(
            route_points,
            offending,
            params.local_router_factory,
            land_index,
            params,
            buffer_nm=buffer_nm,
        )
        if refined is None:
            continue
        route_points = refined

    final_report = validate_route_no_land(route_points, land_index, tolerance_m=params.tolerance_m)
    if not final_report.is_ok:
        raise RuntimeError("Route failed land validation after local refinements")
    return route_points


def _route_points_from_result(result: RouteResult) -> RoutePoints:
    if isinstance(result, tuple):
        return result[0]
    return result


def _buffer_segment_m(segment: BaseGeometry, buffer_m: float) -> BaseGeometry:
    if buffer_m <= 0:
        return segment
    lon, lat = segment.centroid.x, segment.centroid.y
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    projected = shapely_transform(to_local.transform, segment)
    buffered = projected.buffer(buffer_m)
    return shapely_transform(to_wgs84.transform, buffered)


def _merge_bbox(
    existing: Tuple[float, float, float, float] | None,
    new_bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    if existing is None:
        return new_bbox
    minx = min(existing[0], new_bbox[0])
    miny = min(existing[1], new_bbox[1])
    maxx = max(existing[2], new_bbox[2])
    maxy = max(existing[3], new_bbox[3])
    return (minx, miny, maxx, maxy)


def _segments_bbox(route_points: RoutePoints, indices: Sequence[int]) -> Tuple[float, float, float, float]:
    bbox: Tuple[float, float, float, float] | None = None
    for idx in indices:
        segment = LineString([route_points[idx], route_points[idx + 1]])
        bbox = _merge_bbox(bbox, segment.bounds)
    if bbox is None:
        raise ValueError("No segments available for bbox")
    return bbox


def _expand_bbox_nm(bbox: Tuple[float, float, float, float], buffer_nm: float) -> Tuple[float, float, float, float]:
    if buffer_nm <= 0:
        return bbox
    minx, miny, maxx, maxy = bbox
    center_lat = (miny + maxy) / 2.0
    dlon = _nm_to_lon_deg(center_lat, buffer_nm)
    dlat = _nm_to_lat_deg(center_lat, buffer_nm)
    return (minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)


def _expand_bbox_to_points(
    bbox: Tuple[float, float, float, float],
    *points: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    for lon, lat in points:
        minx = min(minx, lon)
        miny = min(miny, lat)
        maxx = max(maxx, lon)
        maxy = max(maxy, lat)
    return (minx, miny, maxx, maxy)


def _nm_to_lon_deg(lat: float, nm: float) -> float:
    lon2, _, _ = _GEOD.fwd(0.0, lat, 90.0, nm * 1852.0)
    dlon = (lon2 - 0.0 + 540.0) % 360.0 - 180.0
    return abs(dlon)


def _nm_to_lat_deg(lat: float, nm: float) -> float:
    _, lat2, _ = _GEOD.fwd(0.0, lat, 0.0, nm * 1852.0)
    return abs(lat2 - lat)


def _build_local_grid(bbox: Tuple[float, float, float, float], resolution_nm: float) -> GridSpec:
    minx, miny, maxx, maxy = bbox
    center_lat = (miny + maxy) / 2.0
    dx = _nm_to_lon_deg(center_lat, resolution_nm)
    dy = _nm_to_lat_deg(center_lat, resolution_nm)
    dx = max(dx, 1e-6)
    dy = max(dy, 1e-6)
    width = max(1, int(math.ceil((maxx - minx) / dx)))
    height = max(1, int(math.ceil((maxy - miny) / dy)))
    return GridSpec(
        crs="EPSG:4326",
        dx=dx,
        dy=dy,
        xmin=minx,
        ymax=maxy,
        width=width,
        height=height,
    )


def _local_land_mask(grid: GridSpec, land_index: LandIndex, params: LandGuardParams) -> np.ndarray:
    cache = _LocalMaskCache(params.local_cache_dir)
    key = cache.key_for(grid, all_touched=False)
    cached = cache.load(key)
    if cached is not None:
        return cached
    bbox_poly = box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    candidates = land_index.tree.query(bbox_poly, predicate="intersects")
    shapes: list[Tuple[object, int]] = []
    for geom in candidates:
        idx = land_index.geom_id_map.get(id(geom))
        if idx is None:
            continue
        geom = land_index.geoms[int(idx)].intersection(bbox_poly)
        if geom.is_empty:
            continue
        shapes.append((geom, 1))

    if not shapes:
        mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    else:
        transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
        mask = features.rasterize(
            shapes,
            out_shape=(grid.height, grid.width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
    cache.save(key, mask, grid)
    return mask


def _choose_anchor_indices(
    route_points: RoutePoints,
    start_seg: int,
    end_seg: int,
    land_index: LandIndex,
    tolerance_m: float,
) -> Tuple[int, int]:
    start_idx = max(0, start_seg - 1)
    end_idx = min(len(route_points) - 1, end_seg + 1)

    while start_idx > 0 and _point_on_land(route_points[start_idx], land_index, tolerance_m):
        start_idx -= 1
    while end_idx < len(route_points) - 1 and _point_on_land(route_points[end_idx], land_index, tolerance_m):
        end_idx += 1

    return start_idx, end_idx


def _point_on_land(point: Tuple[float, float], land_index: LandIndex, tolerance_m: float) -> bool:
    geom = Point(point)
    check_geom = _buffer_segment_m(geom, tolerance_m) if tolerance_m > 0 else geom
    geoms = land_index.tree.query(check_geom, predicate="intersects")
    for geom in geoms:
        idx = land_index.geom_id_map.get(id(geom))
        if idx is None:
            continue
        if land_index.prepared[int(idx)].intersects(check_geom):
            return True
    return False


@dataclass(slots=True)
class _LocalMaskCache:
    cache_dir: Path

    def key_for(self, grid: GridSpec, all_touched: bool) -> str:
        payload = {
            "bbox": [grid.xmin, grid.ymin, grid.xmax, grid.ymax],
            "dx": grid.dx,
            "dy": grid.dy,
            "all_touched": all_touched,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def load(self, key: str) -> np.ndarray | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        loader = MemMapLoader(path, mode="r", dtype=np.uint8)
        return loader.array

    def save(self, key: str, array: np.ndarray, grid: GridSpec) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(key)
        save_memmap(path, array, dtype=np.uint8)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        meta.update(
            {
                "bbox": [grid.xmin, grid.ymin, grid.xmax, grid.ymax],
                "dx": grid.dx,
                "dy": grid.dy,
                "crs": grid.crs,
            }
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"local_land_mask_{key}.npy"
