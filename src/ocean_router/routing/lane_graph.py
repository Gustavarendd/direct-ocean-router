"""Lane-graph routing helpers for TSS macro guidance."""
from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

from ocean_router.core.grid import GridSpec
from ocean_router.data.tss import LaneGraph


def build_lane_graph_macro_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    grid: GridSpec,
    lane_graph: LaneGraph,
    max_snap_nm: float,
) -> Optional[List[Tuple[int, int]]]:
    """Snap to nearest lane nodes and build a macro path on the lane graph."""
    if lane_graph.nodes_lon.size == 0 or lane_graph.edges_u.size == 0:
        return None

    nodes_xy = np.empty((lane_graph.nodes_lon.size, 2), dtype=np.int32)
    for i, (lon, lat) in enumerate(zip(lane_graph.nodes_lon, lane_graph.nodes_lat)):
        nodes_xy[i] = grid.lonlat_to_xy(float(lon), float(lat))

    start_xy = grid.lonlat_to_xy(start[0], start[1])
    end_xy = grid.lonlat_to_xy(end[0], end[1])

    cell_nm = max(grid.dx, grid.dy) * 60.0
    start_idx, start_dist = _nearest_node(nodes_xy, start_xy)
    end_idx, end_dist = _nearest_node(nodes_xy, end_xy)
    if start_dist * cell_nm > max_snap_nm or end_dist * cell_nm > max_snap_nm:
        return None

    path_nodes = _dijkstra_path(
        lane_graph.edges_u,
        lane_graph.edges_v,
        lane_graph.edges_weight,
        start_idx,
        end_idx,
    )
    if path_nodes is None:
        return None

    path_xy: List[Tuple[int, int]] = []
    path_xy.append(start_xy)
    for node_idx in path_nodes:
        node_xy = (int(nodes_xy[node_idx][0]), int(nodes_xy[node_idx][1]))
        if not path_xy or path_xy[-1] != node_xy:
            path_xy.append(node_xy)
    if path_xy[-1] != end_xy:
        path_xy.append(end_xy)
    return path_xy


def lane_graph_mask_window(
    lane_graph: LaneGraph,
    grid: GridSpec,
    x_off: int,
    y_off: int,
    height: int,
    width: int,
    radius_cells: int,
) -> np.ndarray:
    """Rasterize lane-graph edges into a window mask."""
    mask = np.zeros((height, width), dtype=bool)
    if lane_graph.nodes_lon.size == 0 or lane_graph.edges_u.size == 0:
        return mask

    nodes_xy = np.empty((lane_graph.nodes_lon.size, 2), dtype=np.int32)
    for i, (lon, lat) in enumerate(zip(lane_graph.nodes_lon, lane_graph.nodes_lat)):
        nodes_xy[i] = grid.lonlat_to_xy(float(lon), float(lat))

    x_min = x_off - radius_cells
    x_max = x_off + width + radius_cells
    y_min = y_off - radius_cells
    y_max = y_off + height + radius_cells

    for u, v in zip(lane_graph.edges_u, lane_graph.edges_v):
        x0, y0 = nodes_xy[int(u)]
        x1, y1 = nodes_xy[int(v)]
        if max(x0, x1) < x_min or min(x0, x1) > x_max:
            continue
        if max(y0, y1) < y_min or min(y0, y1) > y_max:
            continue
        _draw_line(mask, x0 - x_off, y0 - y_off, x1 - x_off, y1 - y_off)

    if radius_cells > 0:
        from scipy import ndimage

        structure = np.ones((radius_cells * 2 + 1, radius_cells * 2 + 1), dtype=bool)
        mask = ndimage.binary_dilation(mask, structure=structure)
    return mask


def _nearest_node(nodes_xy: np.ndarray, point_xy: Tuple[int, int]) -> Tuple[int, float]:
    dx = nodes_xy[:, 0] - point_xy[0]
    dy = nodes_xy[:, 1] - point_xy[1]
    dist_sq = dx * dx + dy * dy
    idx = int(np.argmin(dist_sq))
    return idx, float(np.sqrt(dist_sq[idx]))


def _dijkstra_path(
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    edges_weight: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Optional[List[int]]:
    max_u = int(edges_u.max()) if edges_u.size else 0
    max_v = int(edges_v.max()) if edges_v.size else 0
    node_count = max(max_u, max_v) + 1
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(node_count)]
    for u, v, w in zip(edges_u, edges_v, edges_weight):
        adjacency[int(u)].append((int(v), float(w)))

    dist = np.full(node_count, np.inf, dtype=np.float64)
    prev = np.full(node_count, -1, dtype=np.int32)
    dist[start_idx] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start_idx)]

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist != dist[u]:
            continue
        if u == end_idx:
            break
        for v, weight in adjacency[u]:
            nd = cur_dist + weight
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[end_idx]):
        return None

    path: List[int] = []
    cur = end_idx
    while cur != -1:
        path.append(cur)
        cur = int(prev[cur])
    path.reverse()
    return path


def _draw_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = mask.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
