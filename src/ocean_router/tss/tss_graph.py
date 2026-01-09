"""Vector-based TSS routing graph utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import json
import math
import pickle
import heapq

import numpy as np
from shapely import wkb
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon, shape
from shapely.ops import split
from shapely.strtree import STRtree

from ocean_router.core.geodesy import angle_diff_deg, bearing_deg, rhumb_distance_nm

TSS_LANE_TYPES = {"separation_lane"}
TSS_ZONE_TYPES = {"separation_zone"}
INLAND_WATERWAYS = {"canal", "river"}


@dataclass(frozen=True)
class TSSGraphConfig:
    sepzone_buffer_nm: float = 0.2
    connector_radius_nm: float = 3.0
    max_connectors: int = 6
    entry_angle_weight: float = 2.0
    entry_max_angle_deg: float = 100.0
    crossing_angle_weight: float = 1.5
    crossing_target_deg: float = 90.0
    crossing_max_angle_deg: float = 70.0
    crossing_penalty: float = 4.0


@dataclass(frozen=True)
class TSSLaneEdge:
    u: int
    v: int
    length_nm: float
    flow_bearing: float
    cost: float
    lane_id: Optional[str]
    geometry: LineString


@dataclass(frozen=True)
class Connector:
    edge_index: int
    point: Tuple[float, float]
    fraction: float
    distance_nm: float
    angle_diff: float
    cost: float
    is_crossing: bool


@dataclass
class TSSLaneGraph:
    nodes: np.ndarray
    edges: list[TSSLaneEdge]
    adjacency: list[list[Tuple[int, int, float]]]
    edge_tree: Optional[STRtree]
    zones: list[Polygon]

    def save(self, path: Path) -> None:
        payload = {
            "nodes": self.nodes,
            "edges_u": np.array([e.u for e in self.edges], dtype=np.int32),
            "edges_v": np.array([e.v for e in self.edges], dtype=np.int32),
            "edges_length_nm": np.array([e.length_nm for e in self.edges], dtype=np.float32),
            "edges_flow_bearing": np.array([e.flow_bearing for e in self.edges], dtype=np.float32),
            "edges_cost": np.array([e.cost for e in self.edges], dtype=np.float32),
            "edges_lane_id": [e.lane_id for e in self.edges],
            "edges_geom_wkb": [e.geometry.wkb for e in self.edges],
            "zones_wkb": [zone.wkb for zone in self.zones],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TSSLaneGraph":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        edges = []
        for u, v, length, flow, cost, lane_id, geom_wkb in zip(
            payload["edges_u"],
            payload["edges_v"],
            payload["edges_length_nm"],
            payload["edges_flow_bearing"],
            payload["edges_cost"],
            payload["edges_lane_id"],
            payload["edges_geom_wkb"],
        ):
            edges.append(
                TSSLaneEdge(
                    u=int(u),
                    v=int(v),
                    length_nm=float(length),
                    flow_bearing=float(flow),
                    cost=float(cost),
                    lane_id=lane_id,
                    geometry=wkb.loads(geom_wkb),
                )
            )
        zones = [wkb.loads(blob) for blob in payload.get("zones_wkb", [])]
        edge_tree = STRtree([edge.geometry for edge in edges]) if edges else None
        adjacency = _build_adjacency(len(payload["nodes"]), edges)
        return cls(
            nodes=payload["nodes"],
            edges=edges,
            adjacency=adjacency,
            edge_tree=edge_tree,
            zones=zones,
        )


def load_tss_geojson(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("features", []))


def filter_ocean_tss_features(
    features: Sequence[dict],
    skip_waterways: Iterable[str] = INLAND_WATERWAYS,
) -> list[dict]:
    skip_set = {w.lower() for w in skip_waterways}
    return [feat for feat in features if not _feature_is_inland(feat, skip_set)]


def build_directed_lane_graph(
    features: Sequence[dict],
    sepzone_buffer_nm: float,
) -> TSSLaneGraph:
    lanes: list[LineString] = []
    lane_meta: list[Tuple[float, Optional[str]]] = []
    zones = build_forbidden_polygons(features, sepzone_buffer_nm)

    for feat in features:
        seamark_type = get_seamark_type(feat)
        if seamark_type not in TSS_LANE_TYPES:
            continue
        flow = _get_flow_bearing(feat)
        if flow is None:
            continue
        is_forward = _get_flow_forward(feat)
        lane_id = _get_lane_id(feat)
        geom = shape(feat.get("geometry"))
        for line in _iter_lines(geom):
            if not is_forward:
                line = LineString(list(line.coords)[::-1])
            lanes.append(line)
            lane_meta.append((flow, lane_id))

    if not lanes:
        return TSSLaneGraph(
            nodes=np.zeros((0, 2), dtype=np.float64),
            edges=[],
            adjacency=[],
            edge_tree=None,
            zones=zones,
        )

    tree = STRtree(lanes)
    zone_tree = STRtree(zones) if zones else None

    nodes: list[Tuple[float, float]] = []
    node_index: dict[Tuple[float, float], int] = {}
    edges: list[TSSLaneEdge] = []

    def _node_id(lon: float, lat: float) -> int:
        key = (round(lon, 7), round(lat, 7))
        if key not in node_index:
            node_index[key] = len(nodes)
            nodes.append((lon, lat))
        return node_index[key]

    for idx, (line, meta) in enumerate(zip(lanes, lane_meta)):
        flow, lane_id = meta
        segments = _split_line(line, idx, lanes, tree)
        for segment in segments:
            if segment.is_empty:
                continue
            seg = _orient_segment(line, segment)
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            if zone_tree is not None:
                hits = zone_tree.query(seg)
                blocked = False
                for zone_idx in hits:
                    if seg.intersects(zones[int(zone_idx)]):
                        blocked = True
                        break
                if blocked:
                    continue

            lon0, lat0 = coords[0]
            lon1, lat1 = coords[-1]
            length_nm = _line_length_nm(seg)
            if length_nm <= 0:
                continue
            seg_bearing = bearing_deg(lon0, lat0, lon1, lat1)
            align = angle_diff_deg(seg_bearing, flow)
            cost = length_nm * (1.0 + align / 90.0)

            u = _node_id(lon0, lat0)
            v = _node_id(lon1, lat1)
            edges.append(
                TSSLaneEdge(
                    u=u,
                    v=v,
                    length_nm=length_nm,
                    flow_bearing=float(flow),
                    cost=float(cost),
                    lane_id=lane_id,
                    geometry=seg,
                )
            )

    nodes_arr = np.array(nodes, dtype=np.float64)
    edge_tree = STRtree([edge.geometry for edge in edges]) if edges else None
    adjacency = _build_adjacency(len(nodes), edges)
    return TSSLaneGraph(
        nodes=nodes_arr,
        edges=edges,
        adjacency=adjacency,
        edge_tree=edge_tree,
        zones=zones,
    )


def build_forbidden_polygons(
    features: Sequence[dict],
    sepzone_buffer_nm: float,
) -> list[Polygon]:
    zones: list[Polygon] = []
    for feat in features:
        seamark_type = get_seamark_type(feat)
        if seamark_type not in TSS_ZONE_TYPES:
            continue
        geom = shape(feat.get("geometry"))
        for zone in _iter_zones(geom, sepzone_buffer_nm):
            zones.append(zone)
    return zones


def make_connectors(
    point: Tuple[float, float],
    tss_graph: TSSLaneGraph,
    forbidden: Sequence[Polygon],
    radius_nm: float,
    max_connectors: int,
    entry_angle_weight: float,
    max_angle_deg: float,
    crossing_target_deg: float,
    crossing_angle_weight: float,
    crossing_max_angle_deg: float,
    crossing_penalty: float,
    bearing_from_lane: bool = False,
) -> list[Connector]:
    if tss_graph.edge_tree is None:
        return []
    lon, lat = point
    ref_point = Point(lon, lat)
    radius_deg = max(radius_nm / 60.0, 1e-6)
    hits = tss_graph.edge_tree.query(ref_point.buffer(radius_deg))
    if hits.size == 0:
        return []

    forbidden_tree = STRtree(list(forbidden)) if forbidden else None
    candidates: list[Connector] = []
    for edge_idx in hits:
        edge = tss_graph.edges[int(edge_idx)]
        geom = edge.geometry
        proj = geom.project(ref_point)
        proj_point = geom.interpolate(proj)
        proj_lon, proj_lat = proj_point.x, proj_point.y
        dist_nm = rhumb_distance_nm(lon, lat, proj_lon, proj_lat)
        if dist_nm > radius_nm:
            continue

        if bearing_from_lane:
            bearing = bearing_deg(proj_lon, proj_lat, lon, lat)
        else:
            bearing = bearing_deg(lon, lat, proj_lon, proj_lat)
        flow = edge.flow_bearing
        angle = angle_diff_deg(bearing, flow)

        entry_cost = dist_nm * (1.0 + entry_angle_weight * (angle / 90.0))

        crossing_angle = abs(angle - crossing_target_deg)
        crossing_ok = crossing_angle <= crossing_max_angle_deg
        crossing_cost = dist_nm * (1.0 + crossing_angle_weight * (crossing_angle / 90.0))
        crossing_cost *= (1.0 + crossing_penalty)

        segment = LineString([ref_point, proj_point])
        if forbidden_tree is not None:
            hits_zone = forbidden_tree.query(segment)
            blocked = False
            for zone_idx in hits_zone:
                if segment.intersects(forbidden[int(zone_idx)]):
                    blocked = True
                    break
            if blocked:
                continue

        if angle <= max_angle_deg:
            candidates.append(
                Connector(
                    edge_index=int(edge_idx),
                    point=(proj_lon, proj_lat),
                    fraction=float(geom.project(proj_point, normalized=True)),
                    distance_nm=dist_nm,
                    angle_diff=angle,
                    cost=entry_cost,
                    is_crossing=False,
                )
            )
        if crossing_ok:
            candidates.append(
                Connector(
                    edge_index=int(edge_idx),
                    point=(proj_lon, proj_lat),
                    fraction=float(geom.project(proj_point, normalized=True)),
                    distance_nm=dist_nm,
                    angle_diff=angle,
                    cost=crossing_cost,
                    is_crossing=True,
                )
            )

    candidates.sort(key=lambda c: c.cost)
    return candidates[:max_connectors]


def route_with_tss(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    open_sea_router: object,
    tss_graph: TSSLaneGraph,
    forbidden: Sequence[Polygon],
    config: TSSGraphConfig,
) -> Tuple[list[Tuple[float, float]], dict]:
    if tss_graph.nodes.size == 0 or not tss_graph.edges:
        path, meta = open_sea_router.route(start, goal)
        return path, meta

    start_connectors = make_connectors(
        start,
        tss_graph,
        forbidden,
        radius_nm=config.connector_radius_nm,
        max_connectors=config.max_connectors,
        entry_angle_weight=config.entry_angle_weight,
        max_angle_deg=config.entry_max_angle_deg,
        crossing_target_deg=config.crossing_target_deg,
        crossing_angle_weight=config.crossing_angle_weight,
        crossing_max_angle_deg=config.crossing_max_angle_deg,
        crossing_penalty=config.crossing_penalty,
    )
    goal_connectors = make_connectors(
        goal,
        tss_graph,
        forbidden,
        radius_nm=config.connector_radius_nm,
        max_connectors=config.max_connectors,
        entry_angle_weight=config.entry_angle_weight,
        max_angle_deg=config.entry_max_angle_deg,
        crossing_target_deg=config.crossing_target_deg,
        crossing_angle_weight=config.crossing_angle_weight,
        crossing_max_angle_deg=config.crossing_max_angle_deg,
        crossing_penalty=config.crossing_penalty,
        bearing_from_lane=True,
    )

    if not start_connectors or not goal_connectors:
        path, meta = open_sea_router.route(start, goal)
        return path, meta

    combined = _build_augmented_graph(tss_graph, start, goal, start_connectors, goal_connectors)
    edge_path = _dijkstra_edges(combined, combined.start_id, combined.end_id)
    if edge_path is None:
        path, meta = open_sea_router.route(start, goal)
        return path, meta

    coords: list[Tuple[float, float]] = []
    for edge_idx in edge_path:
        geom = combined.edges_geom[edge_idx]
        segment_coords = list(geom.coords)
        if not coords:
            coords.extend(segment_coords)
        else:
            if coords[-1] == segment_coords[0]:
                coords.extend(segment_coords[1:])
            else:
                coords.extend(segment_coords)

    return coords, {"tss_used": True, "tss_edges": len(edge_path)}


@dataclass
class _AugmentedGraph:
    nodes: list[Tuple[float, float]]
    edges_u: list[int]
    edges_v: list[int]
    edges_cost: list[float]
    edges_geom: list[LineString]
    adjacency: list[list[Tuple[int, int, float]]]
    start_id: int
    end_id: int


def _build_adjacency(node_count: int, edges: Sequence[TSSLaneEdge]) -> list[list[Tuple[int, int, float]]]:
    adjacency: list[list[Tuple[int, int, float]]] = [[] for _ in range(node_count)]
    for idx, edge in enumerate(edges):
        adjacency[edge.u].append((edge.v, idx, float(edge.cost)))
    return adjacency


def _build_augmented_graph(
    graph: TSSLaneGraph,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    start_connectors: Sequence[Connector],
    goal_connectors: Sequence[Connector],
) -> _AugmentedGraph:
    nodes: list[Tuple[float, float]] = [tuple(pt) for pt in graph.nodes.tolist()]
    edges_u: list[int] = []
    edges_v: list[int] = []
    edges_cost: list[float] = []
    edges_geom: list[LineString] = []
    adjacency: list[list[Tuple[int, int, float]]] = [[] for _ in range(len(nodes))]

    connector_nodes: dict[Tuple[int, float], int] = {}
    frac_eps = 1e-6

    def _ensure_connector_node(edge_index: int, fraction: float, point: Tuple[float, float]) -> int:
        key = (edge_index, round(fraction, 6))
        if key in connector_nodes:
            return connector_nodes[key]
        node_id = len(nodes)
        nodes.append(point)
        adjacency.append([])
        connector_nodes[key] = node_id
        return node_id

    connectors_by_edge: dict[int, list[Tuple[float, int, Tuple[float, float]]]] = {}
    for conn in list(start_connectors) + list(goal_connectors):
        connectors_by_edge.setdefault(conn.edge_index, []).append((conn.fraction, conn.edge_index, conn.point))

    for edge_idx, edge in enumerate(graph.edges):
        u = edge.u
        v = edge.v
        geom = edge.geometry
        if edge_idx not in connectors_by_edge:
            _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, u, v, geom, edge.cost)
            continue

        connector_entries = sorted(connectors_by_edge[edge_idx], key=lambda item: item[0])
        dedup: dict[float, Tuple[float, float]] = {}
        for fraction, _, point in connector_entries:
            fraction = min(max(fraction, 0.0), 1.0)
            if fraction <= 1e-6 or fraction >= 1.0 - 1e-6:
                continue
            dedup[round(fraction, 6)] = point
        fractions = sorted(dedup)
        connector_points = [dedup[fraction] for fraction in fractions]

        if not fractions:
            _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, u, v, geom, edge.cost)
            continue

        split_geom = split(geom, MultiPoint([Point(pt) for pt in connector_points]))
        segments = [seg for seg in split_geom.geoms if isinstance(seg, LineString) and not seg.is_empty]
        if not segments:
            _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, u, v, geom, edge.cost)
            continue

        segments.sort(
            key=lambda seg: geom.project(seg.interpolate(0.5, normalized=True), normalized=True)
        )
        node_ids = [u]
        for fraction, point in zip(fractions, connector_points):
            node_ids.append(_ensure_connector_node(edge_idx, fraction, point))
        node_ids.append(v)
        if len(segments) != len(node_ids) - 1:
            _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, u, v, geom, edge.cost)
            continue

        for seg, n0, n1 in zip(segments, node_ids, node_ids[1:]):
            seg_oriented = _orient_segment(geom, seg)
            seg_length = _line_length_nm(seg_oriented)
            seg_bearing = bearing_deg(seg_oriented.coords[0][0], seg_oriented.coords[0][1], seg_oriented.coords[-1][0], seg_oriented.coords[-1][1])
            align = angle_diff_deg(seg_bearing, edge.flow_bearing)
            seg_cost = seg_length * (1.0 + align / 90.0)
            _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, n0, n1, seg_oriented, seg_cost)

    start_id = len(nodes)
    nodes.append(start)
    adjacency.append([])
    goal_id = len(nodes)
    nodes.append(goal)
    adjacency.append([])

    for conn in start_connectors:
        conn_node = _connector_node_for_edge(graph, conn, frac_eps, _ensure_connector_node)
        geom = LineString([start, conn.point])
        _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, start_id, conn_node, geom, conn.cost)

    for conn in goal_connectors:
        conn_node = _connector_node_for_edge(graph, conn, frac_eps, _ensure_connector_node)
        geom = LineString([conn.point, goal])
        _append_edge(edges_u, edges_v, edges_cost, edges_geom, adjacency, conn_node, goal_id, geom, conn.cost)

    return _AugmentedGraph(
        nodes=nodes,
        edges_u=edges_u,
        edges_v=edges_v,
        edges_cost=edges_cost,
        edges_geom=edges_geom,
        adjacency=adjacency,
        start_id=start_id,
        end_id=goal_id,
    )


def _connector_node_for_edge(
    graph: TSSLaneGraph,
    connector: Connector,
    frac_eps: float,
    ensure_node: callable,
) -> int:
    if connector.fraction <= frac_eps:
        return graph.edges[connector.edge_index].u
    if connector.fraction >= 1.0 - frac_eps:
        return graph.edges[connector.edge_index].v
    return ensure_node(connector.edge_index, connector.fraction, connector.point)


def _append_edge(
    edges_u: list[int],
    edges_v: list[int],
    edges_cost: list[float],
    edges_geom: list[LineString],
    adjacency: list[list[Tuple[int, int, float]]],
    u: int,
    v: int,
    geom: LineString,
    cost: float,
) -> None:
    edge_idx = len(edges_u)
    edges_u.append(u)
    edges_v.append(v)
    edges_cost.append(cost)
    edges_geom.append(geom)
    adjacency[u].append((v, edge_idx, cost))


def _dijkstra_edges(
    graph: _AugmentedGraph,
    start_id: int,
    goal_id: int,
) -> Optional[list[int]]:
    node_count = len(graph.nodes)
    dist = [math.inf] * node_count
    prev_edge = [-1] * node_count
    prev_node = [-1] * node_count
    dist[start_id] = 0.0

    heap: list[Tuple[float, int]] = [(0.0, start_id)]
    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist != dist[u]:
            continue
        if u == goal_id:
            break
        for v, edge_idx, cost in graph.adjacency[u]:
            nd = cur_dist + cost
            if nd < dist[v]:
                dist[v] = nd
                prev_edge[v] = edge_idx
                prev_node[v] = u
                heapq.heappush(heap, (nd, v))

    if not math.isfinite(dist[goal_id]):
        return None

    path_edges: list[int] = []
    cur = goal_id
    while cur != start_id:
        edge_idx = prev_edge[cur]
        if edge_idx < 0:
            break
        path_edges.append(edge_idx)
        cur = prev_node[cur]
    path_edges.reverse()
    return path_edges


def get_seamark_type(feature: dict) -> Optional[str]:
    props = feature.get("properties", {})
    parsed = props.get("parsed_other_tags", {})
    if parsed and "seamark:type" in parsed:
        return parsed["seamark:type"]
    other_tags = props.get("other_tags", "")
    if "seamark:type" in other_tags:
        import re

        match = re.search(r'\"seamark:type\"=>\"([^\"]+)\"', other_tags)
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

            match = re.search(r'\"waterway\"=>\"([^\"]+)\"', other_tags)
            if match:
                waterway = match.group(1)
    if waterway is None:
        return False
    return str(waterway).lower() in skip_waterways


def _iter_lines(geom: object) -> Iterable[LineString]:
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield line


def _iter_zones(geom: object, buffer_nm: float) -> Iterable[Polygon]:
    buffer_deg = buffer_nm / 60.0
    if isinstance(geom, Polygon):
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield poly
    elif geom.geom_type in ("LineString", "MultiLineString"):
        buffered = geom.buffer(buffer_deg)
        if isinstance(buffered, Polygon):
            yield buffered
        elif buffered.geom_type == "MultiPolygon":
            for poly in buffered.geoms:
                yield poly


def _split_line(
    line: LineString,
    idx: int,
    lines: list[LineString],
    tree: STRtree,
) -> list[LineString]:
    points: list[Point] = []
    hits = tree.query(line)
    for hit in hits:
        if int(hit) == idx:
            continue
        other = lines[int(hit)]
        inter = line.intersection(other)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            points.append(inter)
        elif inter.geom_type == "MultiPoint":
            points.extend(list(inter.geoms))
        elif inter.geom_type in ("LineString", "MultiLineString"):
            for geom in getattr(inter, "geoms", [inter]):
                coords = list(geom.coords)
                if coords:
                    points.append(Point(coords[0]))
                    points.append(Point(coords[-1]))
        elif inter.geom_type == "GeometryCollection":
            for geom in inter.geoms:
                if geom.geom_type == "Point":
                    points.append(geom)
                elif geom.geom_type == "MultiPoint":
                    points.extend(list(geom.geoms))
                elif geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    if coords:
                        points.append(Point(coords[0]))
                        points.append(Point(coords[-1]))

    if not points:
        return [line]

    dedup: dict[Tuple[float, float], Point] = {}
    for pt in points:
        key = (round(pt.x, 7), round(pt.y, 7))
        dedup[key] = pt
    splitter = list(dedup.values())
    if not splitter:
        return [line]

    split_points = MultiPoint(splitter)
    split_geom = split(line, split_points)
    segments = [seg for seg in split_geom.geoms if isinstance(seg, LineString) and not seg.is_empty]
    if not segments:
        return [line]
    segments.sort(
        key=lambda seg: line.project(seg.interpolate(0.5, normalized=True), normalized=True)
    )
    return segments


def _orient_segment(base: LineString, segment: LineString) -> LineString:
    coords = list(segment.coords)
    if len(coords) < 2:
        return segment
    start_proj = base.project(Point(coords[0]))
    end_proj = base.project(Point(coords[-1]))
    if end_proj < start_proj:
        return LineString(coords[::-1])
    return segment


def _line_length_nm(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    length = 0.0
    for i in range(len(coords) - 1):
        lon0, lat0 = coords[i]
        lon1, lat1 = coords[i + 1]
        length += rhumb_distance_nm(lon0, lat0, lon1, lat1)
    return length
