"""Hybrid routing helpers to splice vector TSS lanes into grid paths."""
from __future__ import annotations

from typing import List, Tuple

from ocean_router.core.grid import GridSpec
from ocean_router.data.tss import TSSFields
from ocean_router.data.tss_vector import TSSVectorGraph


def refine_path_with_tss_vector(
    path: List[Tuple[float, float]],
    grid: GridSpec,
    tss: TSSFields,
    graph: TSSVectorGraph,
    detection_radius_cells: int,
    connector_radius_nm: float,
    max_connectors: int,
    entry_angle_weight: float,
    entry_max_angle_deg: float,
) -> List[Tuple[float, float]]:
    if not path or tss is None or graph is None:
        return path

    in_tss = []
    grid_points: List[Tuple[int, int] | None] = []
    for lon, lat in path:
        try:
            x, y = grid.lonlat_to_xy(lon, lat)
        except Exception:
            in_tss.append(False)
            grid_points.append(None)
            continue
        if 0 <= y < tss.lane_mask.shape[0] and 0 <= x < tss.lane_mask.shape[1]:
            in_tss.append(tss.in_or_near_lane(y, x, radius=detection_radius_cells))
            grid_points.append((x, y))
        else:
            in_tss.append(False)
            grid_points.append(None)

    # If the segment crosses a lane but points are outside, still mark for refinement.
    for idx in range(len(grid_points) - 1):
        p0 = grid_points[idx]
        p1 = grid_points[idx + 1]
        if p0 is None or p1 is None:
            continue
        if tss.line_intersects_lane(p0[1], p0[0], p1[1], p1[0]):
            in_tss[idx] = True
            in_tss[idx + 1] = True

    segments = _find_true_runs(in_tss)
    if not segments:
        return path

    refined: List[Tuple[float, float]] = []
    cursor = 0
    for start_idx, end_idx in segments:
        if cursor < start_idx:
            refined.extend(path[cursor:start_idx])
        entry_pt = path[start_idx]
        exit_pt = path[end_idx]
        lane_path = graph.route_between_points(
            entry_pt,
            exit_pt,
            connector_radius_nm=connector_radius_nm,
            max_connectors=max_connectors,
            angle_weight=entry_angle_weight,
            max_angle_deg=entry_max_angle_deg,
        )
        if lane_path:
            if refined and lane_path[0] == refined[-1]:
                refined.extend(lane_path[1:])
            else:
                refined.extend(lane_path)
        else:
            refined.extend(path[start_idx:end_idx + 1])
        cursor = end_idx + 1

    if cursor < len(path):
        refined.extend(path[cursor:])
    return refined


def _find_true_runs(flags: List[bool]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start_idx = None
    for idx, flag in enumerate(flags):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            if idx - 1 > start_idx:
                runs.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None and len(flags) - 1 > start_idx:
        runs.append((start_idx, len(flags) - 1))
    return runs
