"""Vector-based TSS lane graph builder and router helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple
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


@dataclass
class ConnectorCandidate:
    edge_index: int
    point: Tuple[float, float]
    fraction: float
    distance_nm: float
    angle_diff: float
    cost: float


@dataclass
class TSSVectorGraph:
    nodes: np.ndarray
    edges_u: np.ndarray
    edges_v: np.ndarray
    edges_length_nm: np.ndarray
    edges_flow_bearing: np.ndarray
    edges_cost: np.ndarray
    edges_lane_id: list[Optional[str]]
    edges_geom: list[LineString]
    zones: list[Polygon]
    edge_tree: Optional[STRtree] = field(init=False, default=None)
    adjacency: list[list[Tuple[int, int, float]]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.edges_geom:
            self.edge_tree = STRtree(self.edges_geom)
        else:
            self.edge_tree = None
        self.adjacency = [[] for _ in range(self.nodes.shape[0])]
        for idx, (u, v, cost) in enumerate(
            zip(self.edges_u, self.edges_v, self.edges_cost)
        ):
            self.adjacency[int(u)].append((int(v), idx, float(cost)))

    def save(self, path: Path) -> None:
        payload = {
            "nodes": self.nodes,
            "edges_u": self.edges_u,
            "edges_v": self.edges_v,
            "edges_length_nm": self.edges_length_nm,
            "edges_flow_bearing": self.edges_flow_bearing,
            "edges_cost": self.edges_cost,
            "edges_lane_id": self.edges_lane_id,
            "edges_geom_wkb": [geom.wkb for geom in self.edges_geom],
            "zones_wkb": [geom.wkb for geom in self.zones],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TSSVectorGraph":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        edges_geom = [wkb.loads(blob) for blob in payload["edges_geom_wkb"]]
        zones = [wkb.loads(blob) for blob in payload["zones_wkb"]]
        return cls(
            nodes=payload["nodes"],
            edges_u=payload["edges_u"],
            edges_v=payload["edges_v"],
            edges_length_nm=payload["edges_length_nm"],
            edges_flow_bearing=payload["edges_flow_bearing"],
            edges_cost=payload["edges_cost"],
            edges_lane_id=payload["edges_lane_id"],
            edges_geom=edges_geom,
            zones=zones,
        )

    def find_connectors(
        self,
        lon: float,
        lat: float,
        radius_nm: float,
        max_candidates: int,
        angle_weight: float,
        max_angle_deg: float,
        bearing_from_lane: bool = False,
    ) -> list[ConnectorCandidate]:
        if self.edge_tree is None:
            return []
        point = Point(lon, lat)
        radius_deg = max(radius_nm / 60.0, 1e-6)
        hits = self.edge_tree.query(point.buffer(radius_deg))
        if hits.size == 0:
            return []

        candidates: list[ConnectorCandidate] = []
        for edge_idx in hits:
            geom = self.edges_geom[int(edge_idx)]
            proj = geom.project(point)
            proj_point = geom.interpolate(proj)
            proj_lon, proj_lat = proj_point.x, proj_point.y
            dist_nm = rhumb_distance_nm(lon, lat, proj_lon, proj_lat)
            if dist_nm > radius_nm:
                continue

            if bearing_from_lane:
                bearing = bearing_deg(proj_lon, proj_lat, lon, lat)
            else:
                bearing = bearing_deg(lon, lat, proj_lon, proj_lat)
            flow = float(self.edges_flow_bearing[int(edge_idx)])
            angle = angle_diff_deg(bearing, flow)
            if angle > max_angle_deg:
                continue

            if self._connector_hits_zone(point, proj_point):
                continue

            cost = dist_nm * (1.0 + angle_weight * (angle / 90.0))
            fraction = geom.project(proj_point, normalized=True)
            candidates.append(
                ConnectorCandidate(
                    edge_index=int(edge_idx),
                    point=(proj_lon, proj_lat),
                    fraction=float(fraction),
                    distance_nm=dist_nm,
                    angle_diff=angle,
                    cost=cost,
                )
            )

        candidates.sort(key=lambda c: c.cost)
        return candidates[:max_candidates]

    def _connector_hits_zone(self, start: Point, end: Point) -> bool:
        if not self.zones:
            return False
        segment = LineString([start, end])
        for zone in self.zones:
            if segment.intersects(zone):
                return True
        return False

    def route_between_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        connector_radius_nm: float,
        max_connectors: int,
        angle_weight: float,
        max_angle_deg: float,
    ) -> Optional[list[Tuple[float, float]]]:
        if self.nodes.size == 0 or not self.edges_geom:
            return None

        start_candidates = self.find_connectors(
            start[0],
            start[1],
            connector_radius_nm,
            max_connectors,
            angle_weight,
            max_angle_deg,
        )
        end_candidates = self.find_connectors(
            end[0],
            end[1],
            connector_radius_nm,
            max_connectors,
            angle_weight,
            max_angle_deg,
            bearing_from_lane=True,
        )
        if not start_candidates or not end_candidates:
            return None

        graph = _build_augmented_graph(
            self,
            start,
            end,
            start_candidates,
            end_candidates,
        )
        edge_path = _dijkstra_edges(graph, graph.start_id, graph.end_id)
        if edge_path is None:
            return None

        coords: list[Tuple[float, float]] = []
        for edge_idx in edge_path:
            geom = graph.edges_geom[edge_idx]
            segment_coords = list(geom.coords)
            if not coords:
                coords.extend(segment_coords)
            else:
                if coords[-1] == segment_coords[0]:
                    coords.extend(segment_coords[1:])
                else:
                    coords.extend(segment_coords)
        return coords if coords else None


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


def load_tss_vector_graph(
    cache_path: Path,
) -> Optional[TSSVectorGraph]:
    if cache_path.exists():
        return TSSVectorGraph.load(cache_path)
    return None


def build_tss_vector_graph(
    geojson_path: Path,
    sepzone_buffer_nm: float,
    skip_waterways: Iterable[str] = INLAND_WATERWAYS,
) -> TSSVectorGraph:
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])

    lanes: list[LineString] = []
    lane_meta: list[Tuple[float, Optional[str]]] = []
    zones: list[Polygon] = []

    for feat in features:
        if _feature_is_inland(feat, skip_waterways):
            continue
        seamark_type = get_seamark_type(feat)
        if seamark_type in TSS_LANE_TYPES:
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
        elif seamark_type in TSS_ZONE_TYPES:
            geom = shape(feat.get("geometry"))
            for zone in _iter_zones(geom, sepzone_buffer_nm):
                zones.append(zone)

    if not lanes:
        return TSSVectorGraph(
            nodes=np.zeros((0, 2), dtype=np.float64),
            edges_u=np.zeros((0,), dtype=np.int32),
            edges_v=np.zeros((0,), dtype=np.int32),
            edges_length_nm=np.zeros((0,), dtype=np.float32),
            edges_flow_bearing=np.zeros((0,), dtype=np.float32),
            edges_cost=np.zeros((0,), dtype=np.float32),
            edges_lane_id=[],
            edges_geom=[],
            zones=zones,
        )

    tree = STRtree(lanes)
    zone_tree = STRtree(zones) if zones else None

    nodes: list[Tuple[float, float]] = []
    node_index: dict[Tuple[float, float], int] = {}
    edges_u: list[int] = []
    edges_v: list[int] = []
    edges_length_nm: list[float] = []
    edges_flow_bearing: list[float] = []
    edges_cost: list[float] = []
    edges_lane_id: list[Optional[str]] = []
    edges_geom: list[LineString] = []

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
            edges_u.append(u)
            edges_v.append(v)
            edges_length_nm.append(length_nm)
            edges_flow_bearing.append(flow)
            edges_cost.append(cost)
            edges_lane_id.append(lane_id)
            edges_geom.append(seg)

    nodes_arr = np.array(nodes, dtype=np.float64)
    return TSSVectorGraph(
        nodes=nodes_arr,
        edges_u=np.array(edges_u, dtype=np.int32),
        edges_v=np.array(edges_v, dtype=np.int32),
        edges_length_nm=np.array(edges_length_nm, dtype=np.float32),
        edges_flow_bearing=np.array(edges_flow_bearing, dtype=np.float32),
        edges_cost=np.array(edges_cost, dtype=np.float32),
        edges_lane_id=edges_lane_id,
        edges_geom=edges_geom,
        zones=zones,
    )


def load_or_build_tss_vector_graph(
    geojson_path: Optional[Path],
    cache_path: Path,
    sepzone_buffer_nm: float,
) -> Optional[TSSVectorGraph]:
    if cache_path.exists():
        if geojson_path is None or not geojson_path.exists():
            return TSSVectorGraph.load(cache_path)
        try:
            if cache_path.stat().st_mtime >= geojson_path.stat().st_mtime:
                return TSSVectorGraph.load(cache_path)
        except OSError:
            pass
    if geojson_path is None or not geojson_path.exists():
        return None
    graph = build_tss_vector_graph(geojson_path, sepzone_buffer_nm)
    graph.save(cache_path)
    return graph


def get_seamark_type(feature: dict) -> Optional[str]:
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


def _build_augmented_graph(
    graph: TSSVectorGraph,
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_connectors: list[ConnectorCandidate],
    end_connectors: list[ConnectorCandidate],
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
    for conn in start_connectors + end_connectors:
        connectors_by_edge.setdefault(conn.edge_index, []).append((conn.fraction, conn.edge_index, conn.point))

    for edge_idx, (u, v, geom, length_nm, flow) in enumerate(
        zip(
            graph.edges_u,
            graph.edges_v,
            graph.edges_geom,
            graph.edges_length_nm,
            graph.edges_flow_bearing,
        )
    ):
        edge_idx = int(edge_idx)
        u = int(u)
        v = int(v)
        if edge_idx not in connectors_by_edge:
            _append_edge(
                edges_u,
                edges_v,
                edges_cost,
                edges_geom,
                adjacency,
                u,
                v,
                geom,
                length_nm,
                flow,
            )
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
            _append_edge(
                edges_u,
                edges_v,
                edges_cost,
                edges_geom,
                adjacency,
                u,
                v,
                geom,
                length_nm,
                flow,
            )
            continue

        split_geom = split(geom, MultiPoint([Point(pt) for pt in connector_points]))
        segments = [seg for seg in split_geom.geoms if isinstance(seg, LineString) and not seg.is_empty]
        if not segments:
            _append_edge(
                edges_u,
                edges_v,
                edges_cost,
                edges_geom,
                adjacency,
                u,
                v,
                geom,
                length_nm,
                flow,
            )
            continue

        segments.sort(
            key=lambda seg: geom.project(seg.interpolate(0.5, normalized=True), normalized=True)
        )
        node_ids = [u]
        for fraction, point in zip(fractions, connector_points):
            node_ids.append(_ensure_connector_node(edge_idx, fraction, point))
        node_ids.append(v)
        if len(segments) != len(node_ids) - 1:
            _append_edge(
                edges_u,
                edges_v,
                edges_cost,
                edges_geom,
                adjacency,
                u,
                v,
                geom,
                length_nm,
                flow,
            )
            continue

        for seg, n0, n1 in zip(segments, node_ids, node_ids[1:]):
            seg_oriented = _orient_segment(geom, seg)
            seg_length = _line_length_nm(seg_oriented)
            _append_edge(
                edges_u,
                edges_v,
                edges_cost,
                edges_geom,
                adjacency,
                n0,
                n1,
                seg_oriented,
                seg_length,
                flow,
            )

    start_id = len(nodes)
    nodes.append(start)
    adjacency.append([])
    end_id = len(nodes)
    nodes.append(end)
    adjacency.append([])

    for conn in start_connectors:
        conn_node = _connector_node_for_edge(graph, conn, frac_eps, _ensure_connector_node)
        geom = LineString([start, conn.point])
        _append_direct_edge(
            edges_u,
            edges_v,
            edges_cost,
            edges_geom,
            adjacency,
            start_id,
            conn_node,
            geom,
            conn.cost,
        )

    for conn in end_connectors:
        conn_node = _connector_node_for_edge(graph, conn, frac_eps, _ensure_connector_node)
        geom = LineString([conn.point, end])
        _append_direct_edge(
            edges_u,
            edges_v,
            edges_cost,
            edges_geom,
            adjacency,
            conn_node,
            end_id,
            geom,
            conn.cost,
        )

    return _AugmentedGraph(
        nodes=nodes,
        edges_u=edges_u,
        edges_v=edges_v,
        edges_cost=edges_cost,
        edges_geom=edges_geom,
        adjacency=adjacency,
        start_id=start_id,
        end_id=end_id,
    )


def _connector_node_for_edge(
    graph: TSSVectorGraph,
    connector: ConnectorCandidate,
    frac_eps: float,
    ensure_node: callable,
) -> int:
    if connector.fraction <= frac_eps:
        return int(graph.edges_u[connector.edge_index])
    if connector.fraction >= 1.0 - frac_eps:
        return int(graph.edges_v[connector.edge_index])
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
    length_nm: float,
    flow_bearing: float,
) -> None:
    coords = list(geom.coords)
    if len(coords) < 2:
        return
    seg_bearing = bearing_deg(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
    align = angle_diff_deg(seg_bearing, flow_bearing)
    cost = length_nm * (1.0 + align / 90.0)
    edge_idx = len(edges_u)
    edges_u.append(u)
    edges_v.append(v)
    edges_cost.append(cost)
    edges_geom.append(geom)
    adjacency[u].append((v, edge_idx, cost))


def _append_direct_edge(
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
    end_id: int,
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
        if u == end_id:
            break
        for v, edge_idx, cost in graph.adjacency[u]:
            nd = cur_dist + cost
            if nd < dist[v]:
                dist[v] = nd
                prev_edge[v] = edge_idx
                prev_node[v] = u
                heapq.heappush(heap, (nd, v))

    if not math.isfinite(dist[end_id]):
        return None

    path_edges: list[int] = []
    cur = end_id
    while cur != start_id:
        edge_idx = prev_edge[cur]
        if edge_idx < 0:
            break
        path_edges.append(edge_idx)
        cur = prev_node[cur]
    path_edges.reverse()
    return path_edges
