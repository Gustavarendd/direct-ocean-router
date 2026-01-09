"""API routers."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import List, Tuple, Optional
from fastapi import APIRouter, Depends, HTTPException

from ocean_router.api.schemas import RouteRequest, RouteResponse
from ocean_router.api.dependencies import (
    get_grid_spec,
    get_bathy,
    get_land_mask,
    get_tss,
    get_cost_weights,
    get_canals,
    get_tss_vector_graph,
)
from ocean_router.core.geodesy import rhumb_interpolate, rhumb_distance_nm
from ocean_router.core.grid import GridSpec
from ocean_router.core.config import get_config
from ocean_router.data.bathy import Bathy
from ocean_router.data.canals import Canal, canal_mask_coarse, canal_mask_window
from ocean_router.data.land import LandMask
from ocean_router.data.tss import TSSFields
from ocean_router.data.tss_vector import TSSVectorGraph
from ocean_router.routing.costs import CostContext, CostWeights
from ocean_router.routing.corridor import build_corridor, build_corridor_with_arrays, precompute_corridor_arrays
from ocean_router.routing.astar import CorridorAStar
from ocean_router.routing.astar_fast import (
    CoarseToFineAStar,
    FastCorridorAStar,
    _build_global_coarse_mask,
    _global_coarse_astar,
    _build_corridor_from_path,
    AStarResult,
)
from ocean_router.routing.astar_numba import FastNumbaCorridorAStar, NUMBA_AVAILABLE
from ocean_router.routing.lane_graph import build_lane_graph_macro_path
from ocean_router.routing.simplify import simplify_path, simplify_between_tss_boundaries, tss_aware_simplify, repair_tss_violations
from ocean_router.routing.tss_hybrid import refine_path_with_tss_vector
from ocean_router.land.land_guard import build_land_index, LandGuardParams, route_with_land_guard


def _is_open_ocean_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    land: Optional[LandMask],
    tss: Optional[TSSFields],
    grid: GridSpec,
) -> bool:
    """Check if the straight-line route stays safely away from land and TSS."""
    if land is None and tss is None:
        return True  # No land/TSS data, assume open ocean
    
    # Sample points along the rhumb line route
    total_dist_nm = rhumb_distance_nm(start[0], start[1], end[0], end[1])
    cell_nm = grid.dy * 60.0
    lat1, lon1 = start[1], start[0]  # start is (lon, lat)
    lat2, lon2 = end[1], end[0]

    # If TSS data exists, disable fast path when the straight line intersects lanes/zones.
    if tss is not None:
        tss_spacing_nm = max(0.25, cell_nm * 0.5)
        tss_samples = max(10, int(total_dist_nm / tss_spacing_nm))
        tss_samples = min(tss_samples, 20000)
        for i in range(tss_samples + 1):
            frac = i / tss_samples
            lon, lat = rhumb_interpolate(lon1, lat1, lon2, lat2, frac)
            try:
                x, y = grid.lonlat_to_xy(lon, lat)
                if not (0 <= y < grid.height and 0 <= x < grid.width):
                    continue
                if (
                    tss.in_sepzone(y, x)
                    or tss.in_sepboundary(y, x)
                    or tss.in_or_near_lane(y, x)
                ):
                    print(
                        f"[TSS CHECK] TSS encountered along straight line at ({lat:.2f}, {lon:.2f}); "
                        "disabling fast path"
                    )
                    return False
            except Exception:
                # If coordinate conversion fails, be conservative
                return False

    if land is None:
        return True  # No land data, assume open ocean

    sample_spacing_nm = max(2.0, cell_nm * 3.0)
    num_samples = max(10, int(total_dist_nm / sample_spacing_nm))
    num_samples = min(num_samples, 5000)
    
    min_land_dist = float('inf')
    for i in range(num_samples + 1):
        # Interpolate position along rhumb line
        frac = i / num_samples
        lon, lat = rhumb_interpolate(lon1, lat1, lon2, lat2, frac)
        
        # Convert to grid coordinates
        try:
            x, y = grid.lonlat_to_xy(lon, lat)
            
            # Check bounds
            if not (0 <= y < grid.height and 0 <= x < grid.width):
                continue  # Out of bounds is OK for open ocean
                
            # Check distance to land
            dist_cells = land.distance_from_land[y, x]
            dist_nm = dist_cells * cell_nm
            min_land_dist = min(min_land_dist, dist_nm)
            
            # For transoceanic routes, require reasonable offshore clearance
            required_dist = 12.0  # Need to be at least 5nm offshore for long routes
            
            if dist_nm < required_dist:
                print(f"[LAND CHECK] Point {i}/{num_samples} at ({lat:.2f}, {lon:.2f}) too close to land: {dist_nm:.2f} nm")
                return False  # Too close to land
        except Exception:
            # If coordinate conversion fails, be conservative
            return False
    
    print(f"[LAND CHECK] Route cleared - min land distance: {min_land_dist:.2f} nm")
    return True  # Route is safely in open ocean

# Cache for CoarseToFineAStar (expensive to initialize)
# We keep two caches - one for short routes (wider corridor) and one for long routes
_coarse_to_fine_cache_short: Optional["CoarseToFineAStar"] = None
_coarse_to_fine_cache_long: Optional["CoarseToFineAStar"] = None

router = APIRouter()

# Use a Numba-backed corridor A* implementation when available (faster inner loop).
CorridorAStarImpl = FastNumbaCorridorAStar if NUMBA_AVAILABLE else FastCorridorAStar


def compute_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    grid: GridSpec,
    bathy: Optional[Bathy],
    land: Optional[LandMask],
    tss: Optional[TSSFields],
    tss_vector: Optional[TSSVectorGraph],
    canals: Optional[List[Canal]],
    weights: CostWeights,
    draft_m: float = 10.0,
    corridor_width_nm: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], float, List[str]]:
    """Compute a route avoiding land and shallow water."""
    global _coarse_to_fine_cache_short, _coarse_to_fine_cache_long
    warnings: List[str] = []
    
    t_start = time.perf_counter()
    
    # Load config
    t0 = time.perf_counter()
    cfg = get_config()
    print(f"[TIMING] Config load: {(time.perf_counter() - t0)*1000:.1f}ms")

    precision_mode = cfg.mode.mode == "precision"
    tss_wrong_way_hard = precision_mode and cfg.mode.precision_wrong_way_hard
    tss_snap_lane_graph = precision_mode and cfg.mode.precision_snap_lane_graph
    tss_disable_lane_smoothing = precision_mode and cfg.mode.precision_disable_lane_smoothing
    if precision_mode:
        weights = replace(
            weights,
            tss_max_lane_deviation_deg=cfg.mode.precision_tss_max_lane_deviation_deg,
        )
    
    # Use config defaults if not specified
    if corridor_width_nm is None:
        corridor_width_nm = cfg.corridor.width_short_nm
    
    # If no preprocessed data, return great circle
    if bathy is None:
        warnings.append("Bathymetry data not loaded; returning rhumb line segment only.")
        distance = rhumb_distance_nm(start[0], start[1], end[0], end[1])
        return [start, end], distance, warnings
    
    # Draft is a positive value (e.g., 10m means vessel needs 10m of water)
    min_draft = draft_m
    
    try:
        result: Optional[AStarResult] = None
        search_time: Optional[float] = None
        astar = None
        # Calculate overall bearing from start to end for TSS filtering
        import math
        lon1, lat1 = start
        lon2, lat2 = end
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        goal_bearing = math.degrees(math.atan2(dlon * math.cos(math.radians((lat1 + lat2) / 2)), dlat)) % 360
        
        # Create cost context
        t0 = time.perf_counter()
        context = CostContext(
            bathy=bathy,
            tss=tss,
            density=None,
            grid_dx=grid.dx,
            grid_dy=grid.dy,
            goal_bearing=goal_bearing,
            land=land,
            grid=grid,
            tss_wrong_way_hard=tss_wrong_way_hard,
            tss_snap_lane_graph=tss_snap_lane_graph,
            tss_disable_lane_smoothing=tss_disable_lane_smoothing,
        )
        print(f"[TIMING] CostContext creation: {(time.perf_counter() - t0)*1000:.1f}ms")
        print(f"[ROUTE] CostContext created with land={'present' if land else 'None'}")

        lane_macro_path: Optional[List[Tuple[int, int]]] = None
        macro_backbone: Optional[List[Tuple[int, int]]] = None
        if (
            tss is not None
            and tss.lane_graph is not None
            and cfg.tss.lane_graph_macro_enabled
        ):
            lane_macro_path = build_lane_graph_macro_path(
                start,
                end,
                grid,
                tss.lane_graph,
                cfg.tss.lane_graph_snap_max_nm,
            )
            if lane_macro_path:
                print(f"[TSS MACRO] Lane-graph macro path with {len(lane_macro_path)} points")
            else:
                print("[TSS MACRO] Lane-graph macro path unavailable")

        # Pre-compute distance transform to enable land proximity penalties during search
        if land is not None:
            t0 = time.perf_counter()
            _ = land.distance_from_land  # Force computation via cached_property access
            print(f"[TIMING] Distance transform access: {(time.perf_counter() - t0)*1000:.1f}ms")

        # Heuristic weight selection (may be adjusted for coarse-scale)
        heuristic_weight = getattr(cfg.algorithm, 'heuristic_weight', 1.0) if hasattr(cfg, 'algorithm') else 1.0
        search_heuristic = heuristic_weight
        try:
            coarse_scale_cfg = int(getattr(cfg.algorithm, 'coarse_scale', cfg.algorithm.coarse_scale))
        except Exception:
            coarse_scale_cfg = getattr(cfg.algorithm, 'coarse_scale', 32)
        # If using coarse-to-fine with a very small coarse scale (e.g., 8)
        # increase the heuristic weight for that search to prune more aggressively
        if coarse_scale_cfg <= 8:
            search_heuristic = max(search_heuristic, 1.5)
        
        # Choose algorithm based on distance and config
        straight_line_dist = rhumb_distance_nm(start[0], start[1], end[0], end[1])
        canal_coarse_override = canal_mask_coarse(canals, grid, cfg.algorithm.coarse_scale) if canals else None
        
        # Fast path for open ocean routes: if stays away from land, return straight line
        if _is_open_ocean_route(start, end, land, tss, grid):
            print(f"[FAST PATH] Open ocean route detected ({straight_line_dist:.1f} nm), returning straight line")
            return [start, end], rhumb_distance_nm(start[0], start[1], end[0], end[1]), []
        
        use_coarse_to_fine = (
            cfg.algorithm.default == "coarse_to_fine" or
            (cfg.algorithm.default == "auto" and 
             straight_line_dist > cfg.algorithm.coarse_to_fine_threshold_nm)
        )

        t0 = time.perf_counter()
        def _macro_guided_corridor() -> Optional[AStarResult]:
            nonlocal macro_backbone
            try:
                scale_try = int(getattr(cfg.algorithm, 'coarse_scale', 8))
                land_mask_macro = land.base if land else None
                canal_override = canal_mask_coarse(canals, grid, scale_try) if canals else None
                macro_mask = _build_global_coarse_mask(
                    bathy.depth,
                    land_mask_macro,
                    min_draft,
                    scale_try,
                    override_mask=canal_override,
                )
                sx, sy = grid.lonlat_to_xy(start[0], start[1])
                gx, gy = grid.lonlat_to_xy(end[0], end[1])
                macro_path = _global_coarse_astar(
                    macro_mask,
                    (sx, sy),
                    (gx, gy),
                    scale_try,
                    grid.dx,
                    grid.dy,
                    grid.ymax,
                    bathy_depth=bathy.depth,
                    land_mask=land_mask_macro,
                    min_draft=min_draft,
                )
                if macro_path is None:
                    return None

                print(f"[FAST PATH] Land-aware macro route found ({len(macro_path)} points), building corridor")
                cell_nm = max(grid.dx * 60.0, 1e-6)
                width_cells = max(2, int(round(corridor_width_nm / cell_nm)))
                full_macro = [(sx, sy)] + macro_path + [(gx, gy)]
                macro_backbone = full_macro
                corridor_mask, x_off, y_off = _build_corridor_from_path(
                    full_macro,
                    (grid.height, grid.width),
                    land.base if land else None,
                    width_cells,
                )
                # Ensure start/end are inside corridor
                sx_local = sx - x_off
                sy_local = sy - y_off
                gx_local = gx - x_off
                gy_local = gy - y_off
                if 0 <= sy_local < corridor_mask.shape[0] and 0 <= sx_local < corridor_mask.shape[1]:
                    corridor_mask[sy_local, sx_local] = 1
                if 0 <= gy_local < corridor_mask.shape[0] and 0 <= gx_local < corridor_mask.shape[1]:
                    corridor_mask[gy_local, gx_local] = 1

                override_mask = None
                if canals:
                    override_mask = canal_mask_window(canals, grid, x_off, y_off, corridor_mask.shape[0], corridor_mask.shape[1])
                    if override_mask is not None and override_mask.size:
                        corridor_mask[override_mask > 0] = 1

                pre = precompute_corridor_arrays(
                    corridor_mask,
                    x_off,
                    y_off,
                    context=context,
                    min_draft=min_draft,
                    weights=weights,
                    override_mask=override_mask,
                    corridor_path=full_macro,
                    grid=grid,
                )
                macro_astar = CorridorAStarImpl(grid, corridor_mask, x_off, y_off, precomputed=pre)
                macro_res = macro_astar.search(start, end, context, weights, min_draft, heuristic_weight=search_heuristic)
                return macro_res if macro_res.success else None
            except Exception as e:
                print(f"[FAST PATH] Macro-guided corridor failed: {e}")
                return None

        if use_coarse_to_fine:
            # Use coarse-to-fine A* for long-distance routes
            # Use wider corridor for medium routes (may pass through straits like Florida)
            # Use narrower corridor for very long routes (mostly open ocean)
            if straight_line_dist > cfg.corridor.long_route_threshold_nm:
                # Long route - use narrow corridor
                if _coarse_to_fine_cache_long is None:
                    print(f"[TIMING] Initializing CoarseToFineAStar (long)...")
                    t_init = time.perf_counter()
                    _coarse_to_fine_cache_long = CoarseToFineAStar(
                        grid,
                        bathy.depth,
                        land_mask=land.base if land else None,
                        coarse_scale=cfg.algorithm.coarse_scale,
                        corridor_width_nm=cfg.corridor.width_long_nm,
                        coarse_override=canal_coarse_override,
                        canals=canals,
                    )
                    print(f"[TIMING] CoarseToFineAStar init (long): {(time.perf_counter() - t_init)*1000:.1f}ms")
                astar = _coarse_to_fine_cache_long
            else:
                # Medium route - use wider corridor for straits
                if _coarse_to_fine_cache_short is None:
                    print(f"[TIMING] Initializing CoarseToFineAStar (short)...")
                    t_init = time.perf_counter()
                    _coarse_to_fine_cache_short = CoarseToFineAStar(
                        grid,
                        bathy.depth,
                        land_mask=land.base if land else None,
                        coarse_scale=cfg.algorithm.coarse_scale,
                        corridor_width_nm=cfg.corridor.width_short_nm,
                        coarse_override=canal_coarse_override,
                        canals=canals,
                    )
                    print(f"[TIMING] CoarseToFineAStar init (short): {(time.perf_counter() - t_init)*1000:.1f}ms")
                astar = _coarse_to_fine_cache_short
        else:
            # Try a global coarse-to-fine pass first (no corridor) so we always
            # attempt a no-corridor macro-route before constructing a corridor.
            print("[TIMING] Attempting global coarse-to-fine pass (no corridor)...")
            try:
                result = _macro_guided_corridor()
                if result is None:
                    # First try a permissive global coarse macro-route (ignore land/depth)
                    scale_try = int(getattr(cfg.algorithm, 'coarse_scale', 8))
                    try:
                        perm_coarse_mask = _build_global_coarse_mask(
                            bathy.depth,
                            None,  # ignore land for permissive macro
                            min_draft=0.0,
                            scale=scale_try,
                        )
                        sx, sy = grid.lonlat_to_xy(start[0], start[1])
                        gx, gy = grid.lonlat_to_xy(end[0], end[1])
                        macro_perm = _global_coarse_astar(
                            perm_coarse_mask,
                            (sx, sy),
                            (gx, gy),
                            scale_try,
                            grid.dx,
                            grid.dy,
                            grid.ymax,
                            bathy_depth=None,
                            land_mask=None,
                            min_draft=0.0,
                        )
                    except Exception:
                        macro_perm = None

                    if macro_perm is not None:
                        # Try to stitch the permissive macro-route using fine A* segments
                        try:
                            print(f"[FALLBACK] Permissive macro-route found ({len(macro_perm)} points), attempting stitching")
                            full_lonlats = []
                            total_expl = 0
                            total_cost_seg = 0.0
                            for i in range(len(macro_perm) - 1):
                                p0 = macro_perm[i]
                                p1 = macro_perm[i + 1]
                                lon0, lat0 = grid.xy_to_lonlat(p0[0], p0[1])
                                lon1, lat1 = grid.xy_to_lonlat(p1[0], p1[1])
                                seg_corridor, seg_x_off, seg_y_off, seg_pre = build_corridor_with_arrays(
                                    grid, (lon0, lat0), (lon1, lat1),
                                    land_mask=land.base if land else None,
                                    width_nm=corridor_width_nm,
                                    context=context,
                                    min_draft=min_draft,
                                    weights=weights,
                                    canals=canals,
                                )
                                seg_astar = CorridorAStarImpl(grid, seg_corridor, seg_x_off, seg_y_off, precomputed=seg_pre)
                                seg_res = seg_astar.search((lon0, lat0), (lon1, lat1), context, weights, min_draft, heuristic_weight=search_heuristic)
                                total_expl += seg_res.explored
                                total_cost_seg += seg_res.cost if seg_res.success else 0.0
                                if not seg_res.success:
                                    raise RuntimeError("perm macro segment failed")
                                if not full_lonlats:
                                    full_lonlats.extend(seg_res.path)
                                else:
                                    full_lonlats.extend(seg_res.path[1:])

                            result = AStarResult(path=full_lonlats, explored=total_expl, cost=total_cost_seg, success=True)
                            print("[FALLBACK] Permissive macro-route stitching succeeded; using stitched path")
                        except Exception:
                            macro_perm = None

                    if macro_perm is None:
                        # Use existing caches if available, prefer short cache
                        if _coarse_to_fine_cache_short is None:
                            _coarse_to_fine_cache_short = CoarseToFineAStar(
                                grid,
                                bathy.depth,
                                land_mask=land.base if land else None,
                                coarse_scale=cfg.algorithm.coarse_scale,
                                corridor_width_nm=cfg.corridor.width_short_nm,
                                coarse_override=canal_coarse_override,
                                canals=canals,
                            )
                        coarse_astar_try = _coarse_to_fine_cache_short
                        t_coarse_try = time.perf_counter()
                        coarse_try_res = coarse_astar_try.search(start, end, context, weights, min_draft, heuristic_weight=search_heuristic)
                        t_coarse_try = time.perf_counter() - t_coarse_try
                        print(f"[TIMING] Coarse-to-fine trial: {t_coarse_try*1000:.1f}ms ({t_coarse_try:.2f}s)")
                        if coarse_try_res.success:
                            print("[FAST PATH] Coarse-to-fine trial succeeded without corridor")
                            result = coarse_try_res
                            search_time = t_coarse_try
                        else:
                            print("[TIMING] Building corridor...")
                            corridor_backbone = macro_backbone or lane_macro_path
                            corridor_mask, x_off, y_off, pre = build_corridor_with_arrays(
                                grid, start, end,
                                land_mask=land.base if land else None,
                                width_nm=corridor_width_nm,
                                context=context,
                                min_draft=min_draft,
                                weights=weights,
                                canals=canals,
                                corridor_path=corridor_backbone,
                                corridor_backbone=corridor_backbone,
                            )
                            astar = CorridorAStarImpl(grid, corridor_mask, x_off, y_off, precomputed=pre)
            except Exception as e:
                print(f"[TIMING] Coarse-to-fine trial error: {e}; falling back to corridor build")
                print("[TIMING] Building corridor...")
                corridor_backbone = macro_backbone or lane_macro_path
                corridor_mask, x_off, y_off, pre = build_corridor_with_arrays(
                    grid, start, end,
                    land_mask=land.base if land else None,
                    width_nm=corridor_width_nm,
                    context=context,
                    min_draft=min_draft,
                    weights=weights,
                    canals=canals,
                    corridor_path=corridor_backbone,
                    corridor_backbone=corridor_backbone,
                )
                astar = CorridorAStarImpl(grid, corridor_mask, x_off, y_off, precomputed=pre)
        print(f"[TIMING] Algorithm setup: {(time.perf_counter() - t0)*1000:.1f}ms")
        
        if result is None:
            t0 = time.perf_counter()
            result = astar.search(start, end, context, weights, min_draft, heuristic_weight=search_heuristic)
            t_search = time.perf_counter() - t0
            print(f"[TIMING] A* search: {t_search*1000:.1f}ms ({t_search:.2f}s)")
            search_time = t_search

        # If coarse-to-fine A* failed, try a macro-guided corridor before corridor fallback
        if not result.success:
            if use_coarse_to_fine:
                macro_res = _macro_guided_corridor()
                if macro_res is not None:
                    print("[FALLBACK] Macro-guided corridor succeeded; using macro path")
                    result = macro_res

        # If still failing, try a corridor-based fallback before giving up
        if not result.success:
            print("[FALLBACK] Coarse-to-fine A* failed; trying corridor-based A*")
            try:
                corridor_backbone = macro_backbone or lane_macro_path
                corridor_mask, x_off, y_off, pre = build_corridor_with_arrays(
                    grid, start, end,
                    land_mask=land.base if land else None,
                    width_nm=corridor_width_nm,
                    context=context,
                    min_draft=min_draft,
                    weights=weights,
                    canals=canals,
                    corridor_path=corridor_backbone,
                    corridor_backbone=corridor_backbone,
                )
                fallback_astar = CorridorAStarImpl(grid, corridor_mask, x_off, y_off, precomputed=pre)
                t0_fb = time.perf_counter()
                fb_result = fallback_astar.search(start, end, context, weights, min_draft, heuristic_weight=heuristic_weight)
                t_fb = time.perf_counter() - t0_fb
                print(f"[TIMING] Fallback A* search: {t_fb*1000:.1f}ms ({t_fb:.2f}s)")
                if fb_result.success:
                    print("[FALLBACK] Corridor A* succeeded; using fallback path")
                    result = fb_result
                    search_time = t_fb
                else:
                    print("[FALLBACK] Corridor A* also failed")
            except Exception as e:
                print(f"[FALLBACK] Corridor fallback error: {e}")

            # Progressive expansion: try wider corridors before giving up
            if not result.success:
                try:
                    for factor in (1.5, 2.0, 4.0):
                        expanded_width = corridor_width_nm * factor
                        print(f"[FALLBACK] Trying expanded corridor width: {expanded_width} nm (factor={factor})")
                        corridor_backbone = macro_backbone or lane_macro_path
                        corridor_mask, x_off, y_off, pre_e = build_corridor_with_arrays(
                            grid, start, end,
                            land_mask=land.base if land else None,
                            width_nm=expanded_width,
                            context=context,
                            min_draft=min_draft,
                            weights=weights,
                            canals=canals,
                            corridor_path=corridor_backbone,
                            corridor_backbone=corridor_backbone,
                        )
                        expanded_astar = CorridorAStarImpl(grid, corridor_mask, x_off, y_off, precomputed=pre_e)
                        t0_e = time.perf_counter()
                        expanded_heuristic = min(3.0, max(search_heuristic, heuristic_weight * factor))
                        e_res = expanded_astar.search(start, end, context, weights, min_draft, heuristic_weight=expanded_heuristic)
                        t_e = time.perf_counter() - t0_e
                        print(f"[TIMING] Expanded corridor A* ({factor}x) search: {t_e*1000:.1f}ms ({t_e:.2f}s)")
                        if e_res.success:
                            print(f"[FALLBACK] Expanded corridor {factor}x succeeded; using expanded path")
                            result = e_res
                            search_time = t_e
                            break
                except Exception as e:
                    print(f"[FALLBACK] Expanded corridor error: {e}")
            # If still failing, try relaxed-context searches (disable TSS/land progressively)
            if not result.success:
                try:
                    # 1) Disable TSS (ignore TSS blocking/penalties)
                    print("[FALLBACK] Trying relaxed search: disable TSS")
                    relaxed_ctx = CostContext(
                        bathy=context.bathy,
                        tss=None,
                        density=context.density,
                        grid_dx=context.grid_dx,
                        grid_dy=context.grid_dy,
                        goal_bearing=context.goal_bearing,
                        land=context.land,
                        grid=grid,
                        tss_wrong_way_hard=False,
                        tss_snap_lane_graph=False,
                        tss_disable_lane_smoothing=False,
                    )
                    t0_r = time.perf_counter()
                    corridor_backbone = macro_backbone or lane_macro_path
                    r_corridor, r_x_off, r_y_off, r_pre = build_corridor_with_arrays(
                        grid, start, end,
                        land_mask=land.base if land else None,
                        width_nm=corridor_width_nm,
                        context=relaxed_ctx,
                        min_draft=min_draft,
                        weights=weights,
                        canals=canals,
                        corridor_path=corridor_backbone,
                        corridor_backbone=corridor_backbone,
                    )
                    r_res = CorridorAStarImpl(grid, r_corridor, r_x_off, r_y_off, precomputed=r_pre).search(start, end, relaxed_ctx, weights, min_draft, heuristic_weight=heuristic_weight)
                    t_r = time.perf_counter() - t0_r
                    print(f"[TIMING] Relaxed (no TSS) A* search: {t_r*1000:.1f}ms ({t_r:.2f}s)")
                    if r_res.success:
                        print("[FALLBACK] Relaxed (no TSS) search succeeded; using relaxed path")
                        result = r_res
                        search_time = t_r
                except Exception as e:
                    print(f"[FALLBACK] Relaxed (no TSS) error: {e}")

            if not result.success:
                try:
                    # 2) Disable land proximity checks (set land to None)
                    print("[FALLBACK] Trying relaxed search: disable land proximity")
                    relaxed_ctx2 = CostContext(
                        bathy=context.bathy,
                        tss=context.tss,
                        density=context.density,
                        grid_dx=context.grid_dx,
                        grid_dy=context.grid_dy,
                        goal_bearing=context.goal_bearing,
                        land=None,
                        grid=grid,
                        tss_wrong_way_hard=tss_wrong_way_hard,
                        tss_snap_lane_graph=tss_snap_lane_graph,
                        tss_disable_lane_smoothing=tss_disable_lane_smoothing,
                    )
                    t0_r2 = time.perf_counter()
                    corridor_backbone = macro_backbone or lane_macro_path
                    r2_corridor, r2_x_off, r2_y_off, r2_pre = build_corridor_with_arrays(
                        grid, start, end,
                        land_mask=None,
                        width_nm=corridor_width_nm,
                        context=relaxed_ctx2,
                        min_draft=min_draft,
                        weights=weights,
                        canals=canals,
                        corridor_path=corridor_backbone,
                        corridor_backbone=corridor_backbone,
                    )
                    r2_res = CorridorAStarImpl(grid, r2_corridor, r2_x_off, r2_y_off, precomputed=r2_pre).search(start, end, relaxed_ctx2, weights, min_draft, heuristic_weight=heuristic_weight)
                    t_r2 = time.perf_counter() - t0_r2
                    print(f"[TIMING] Relaxed (no land) A* search: {t_r2*1000:.1f}ms ({t_r2:.2f}s)")
                    if r2_res.success:
                        print("[FALLBACK] Relaxed (no land) search succeeded; using relaxed path")
                        result = r2_res
                        search_time = t_r2
                except Exception as e:
                    print(f"[FALLBACK] Relaxed (no land) error: {e}")

            if not result.success:
                try:
                    # 3) Disable both TSS and land
                    print("[FALLBACK] Trying relaxed search: disable TSS and land")
                    relaxed_ctx3 = CostContext(
                        bathy=context.bathy,
                        tss=None,
                        density=context.density,
                        grid_dx=context.grid_dx,
                        grid_dy=context.grid_dy,
                        goal_bearing=context.goal_bearing,
                        land=None,
                        grid=grid,
                        tss_wrong_way_hard=False,
                        tss_snap_lane_graph=False,
                        tss_disable_lane_smoothing=False,
                    )
                    t0_r3 = time.perf_counter()
                    corridor_backbone = macro_backbone or lane_macro_path
                    r3_corridor, r3_x_off, r3_y_off, r3_pre = build_corridor_with_arrays(
                        grid, start, end,
                        land_mask=None,
                        width_nm=corridor_width_nm,
                        context=relaxed_ctx3,
                        min_draft=min_draft,
                        weights=weights,
                        canals=canals,
                        corridor_path=corridor_backbone,
                        corridor_backbone=corridor_backbone,
                    )
                    r3_res = CorridorAStarImpl(grid, r3_corridor, r3_x_off, r3_y_off, precomputed=r3_pre).search(start, end, relaxed_ctx3, weights, min_draft, heuristic_weight=heuristic_weight)
                    t_r3 = time.perf_counter() - t0_r3
                    print(f"[TIMING] Relaxed (no TSS/no land) A* search: {t_r3*1000:.1f}ms ({t_r3:.2f}s)")
                    if r3_res.success:
                        print("[FALLBACK] Relaxed (no TSS/no land) search succeeded; using relaxed path")
                        result = r3_res
                        search_time = t_r3
                except Exception as e:
                    print(f"[FALLBACK] Relaxed (no TSS/no land) error: {e}")

            # Global coarse fallback: try a very coarse A* to get a macro-route,
            # then stitch fine corridor segments along that macro-route.
            if not result.success:
                try:
                    print("[FALLBACK] Trying global-coarse macro-route fallback")
                    # Choose a very coarse scale (large cells) to get a fast macro path
                    scale_global = max(64, int(getattr(cfg.algorithm, 'coarse_scale', 8)) * 8)
                    coarse_mask = _build_global_coarse_mask(
                        bathy.depth,
                        land.base if land else None,
                        min_draft,
                        scale_global,
                        override_mask=canal_mask_coarse(canals, grid, scale_global) if canals else None,
                    )
                    sx, sy = grid.lonlat_to_xy(start[0], start[1])
                    gx, gy = grid.lonlat_to_xy(end[0], end[1])
                    macro = _global_coarse_astar(
                        coarse_mask,
                        (sx, sy),
                        (gx, gy),
                        scale_global,
                        grid.dx,
                        grid.dy,
                        grid.ymax,
                        bathy.depth,
                        land.base if land else None,
                        min_draft
                    )
                    if macro is not None:
                        print(f"[FALLBACK] Global coarse macro-route found ({len(macro)} points), stitching segments")
                        full_lonlats: List[Tuple[float, float]] = []
                        total_explored = 0
                        total_cost = 0.0
                        for i in range(len(macro) - 1):
                            p0 = macro[i]
                            p1 = macro[i + 1]
                            lon0, lat0 = grid.xy_to_lonlat(p0[0], p0[1])
                            lon1, lat1 = grid.xy_to_lonlat(p1[0], p1[1])
                            # Build a narrow corridor for the macro segment
                            seg_corridor, seg_x_off, seg_y_off, seg_pre = build_corridor_with_arrays(
                                grid, (lon0, lat0), (lon1, lat1),
                                land_mask=land.base if land else None,
                                width_nm=cfg.corridor.width_short_nm,
                                context=context,
                                min_draft=min_draft,
                                weights=weights,
                                canals=canals,
                            )
                            seg_astar = CorridorAStarImpl(grid, seg_corridor, seg_x_off, seg_y_off, precomputed=seg_pre)
                            seg_res = seg_astar.search((lon0, lat0), (lon1, lat1), context, weights, min_draft, heuristic_weight=heuristic_weight)
                            total_explored += seg_res.explored
                            total_cost += seg_res.cost if seg_res.success else 0.0
                            if not seg_res.success:
                                raise RuntimeError("macro segment failed; aborting global-coarse fallback")
                            if not full_lonlats:
                                full_lonlats.extend(seg_res.path)
                            else:
                                full_lonlats.extend(seg_res.path[1:])

                        result = AStarResult(path=full_lonlats, explored=total_explored, cost=total_cost, success=True)
                        print("[FALLBACK] Global-coarse stitching succeeded")
                    else:
                        print("[FALLBACK] Global-coarse did not find a macro-route")
                except Exception as e:
                    print(f"[FALLBACK] Global-coarse fallback error: {e}")
        
        print(f"[ROUTE RESULT] explored={result.explored}, cost={result.cost:.1f}nm, path_len={len(result.path)}")
        
        # Check final path for TSS violations
        t0 = time.perf_counter()
        if tss and result.success:
            violations = 0
            for i, (lon, lat) in enumerate(result.path):
                x, y = grid.lonlat_to_xy(lon, lat)
                if tss.in_lane(y, x):
                    # Check direction if we have a prev point
                    if i > 0:
                        prev_lon, prev_lat = result.path[i-1]
                        import math
                        from ocean_router.core.geodesy import angle_diff_deg
                        bearing = math.degrees(math.atan2(lon - prev_lon, lat - prev_lat)) % 360
                        preferred = float(tss.direction_field[y, x])
                        angle = angle_diff_deg(bearing, preferred)
                        if angle > 90:
                            violations += 1
                            print(f"[FINAL PATH VIOLATION] point {i}: bearing={bearing:.1f}, preferred={preferred:.1f}, angle_diff={angle:.1f}")
            if violations > 0:
                print(f"[FINAL PATH] Total TSS wrong-way violations: {violations}")
        print(f"[TIMING] TSS violation check: {(time.perf_counter() - t0)*1000:.1f}ms")

        if tss and tss_vector and cfg.tss.vector_graph_enabled and result.success:
            t0 = time.perf_counter()
            refined = refine_path_with_tss_vector(
                result.path,
                grid,
                tss,
                tss_vector,
                detection_radius_cells=cfg.tss.vector_graph_detection_radius_cells,
                connector_radius_nm=cfg.tss.vector_graph_connector_radius_nm,
                max_connectors=cfg.tss.vector_graph_max_connectors,
                entry_angle_weight=cfg.tss.vector_graph_entry_angle_weight,
                entry_max_angle_deg=cfg.tss.vector_graph_entry_max_angle_deg,
            )
            if refined != result.path:
                print(f"[TSS VECTOR] Replaced path segments using lane graph ({len(result.path)} -> {len(refined)} points)")
                result.path = refined
            print(f"[TIMING] TSS vector refine: {(time.perf_counter() - t0)*1000:.1f}ms")

        if not result.success:
            warnings.append("A* search did not find a valid path; returning rhumb line.")
            distance = rhumb_distance_nm(start[0], start[1], end[0], end[1])
            print(f"[TIMING] Total route computation: {(time.perf_counter() - t_start)*1000:.1f}ms")
            return [start, end], distance, warnings
        
        # TSS-aware path simplification:
        # 1. Collapse TSS-to-TSS spans if a straight water path exists
        # 2. Try to skip intermediate TSS runs (A -> B -> C => go A -> C if clear)
        # 3. In non-TSS stretches, greedily remove intermediates when straight path is clear
        # Convert minimum land distance to cells based on grid resolution
        t0 = time.perf_counter()
        min_land_dist_cells = max(2, int(cfg.land.min_distance_nm / (grid.dx * 60)))
        # Increase simplification aggressiveness for long open-ocean voyages
        adaptive_max_simplify = max(cfg.simplify.max_simplify_nm, min(1000.0, straight_line_dist * 0.5))
        path, in_tss = tss_aware_simplify(
            result.path,
            grid,
            tss,
            land,
            preserve_points=None,
            max_simplify_nm=adaptive_max_simplify,
            min_land_distance_cells=min_land_dist_cells,
            snap_to_lane_graph=tss_snap_lane_graph,
            disable_lane_smoothing=tss_disable_lane_smoothing,
        )
        t_simplify = time.perf_counter() - t0
        print(f"[SIMPLIFY] {len(result.path)} -> {len(path)} waypoints")
        print(f"[TIMING] Path simplification: {t_simplify*1000:.1f}ms ({t_simplify:.2f}s)")
        
        # Repair any TSS violations in the path (wrong-way segments)
        t0 = time.perf_counter()
        if tss is not None:
            path, num_repairs = repair_tss_violations(
                path, grid, tss, land, min_land_dist_cells,
                bypass_offsets_nm=cfg.bypass.offset_distances_nm
            )
            if num_repairs > 0:
                print(f"[TSS REPAIR] Made {num_repairs} repairs to avoid wrong-way TSS segments")
        print(f"[TIMING] TSS repair: {(time.perf_counter() - t0)*1000:.1f}ms")
        
        # Calculate total distance
        t0 = time.perf_counter()
        total_dist = 0.0
        for i in range(len(path) - 1):
            total_dist += rhumb_distance_nm(
                path[i][0], path[i][1],
                path[i+1][0], path[i+1][1]
            )
        print(f"[TIMING] Distance calculation: {(time.perf_counter() - t0)*1000:.1f}ms")

        if search_time is not None and total_dist > 0:
            per_1000 = total_dist / 1000.0
            nodes_per_1000 = result.explored / per_1000
            time_per_1000 = search_time / per_1000
            print(f"[BENCH] A* explored={result.explored}, nodes/1000nm={nodes_per_1000:.1f}, time/1000nm={time_per_1000:.2f}s")
        
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] ===== TOTAL ROUTE COMPUTATION: {t_total*1000:.1f}ms ({t_total:.2f}s) =====")
        
        return path, total_dist, warnings
        
    except Exception as e:
        warnings.append(f"Routing error: {str(e)}; returning rhumb line segment.")
        distance = rhumb_distance_nm(start[0], start[1], end[0], end[1])
        return [start, end], distance, warnings


@router.post("/route", response_model=RouteResponse)
def route(
    req: RouteRequest,
    grid: GridSpec = Depends(get_grid_spec),
    bathy: Optional[Bathy] = Depends(get_bathy),
    land: Optional[LandMask] = Depends(get_land_mask),
    tss: Optional[TSSFields] = Depends(get_tss),
    tss_vector: Optional[TSSVectorGraph] = Depends(get_tss_vector_graph),
    canals: List[Canal] = Depends(get_canals),
    weights: CostWeights = Depends(get_cost_weights),
) -> RouteResponse:
    """Compute a ship route between waypoints or two points.
    
    Coordinates are expected as [lat, lon] format.
    """
    t_endpoint_start = time.perf_counter()
    print(f"\n[TIMING] ========== ROUTE REQUEST START ==========")
    
    cfg = get_config()
    precision_mode = cfg.mode.mode == "precision"
    tss_snap_lane_graph = precision_mode and cfg.mode.precision_snap_lane_graph
    tss_disable_lane_smoothing = precision_mode and cfg.mode.precision_disable_lane_smoothing
    
    # Convert input from [lat, lon] to internal [lon, lat] format
    t0 = time.perf_counter()
    if req.waypoints:
        if len(req.waypoints) < 2:
            raise HTTPException(status_code=400, detail="At least two waypoints are required.")
        lonlat_waypoints = [(lon, lat) for lat, lon in req.waypoints]
    else:
        if req.start is None or req.end is None:
            raise HTTPException(status_code=400, detail="Either waypoints or start/end must be provided.")
        lonlat_waypoints = [(req.start[1], req.start[0]), (req.end[1], req.end[0])]
    draft_m = req.draft_m if req.draft_m else 10.0
    corridor_width = req.corridor_width_nm if req.corridor_width_nm else cfg.corridor.width_short_nm
    print(f"[TIMING] Request parsing: {(time.perf_counter() - t0)*1000:.1f}ms")

    warnings: List[str] = []
    if len(lonlat_waypoints) == 2:
        path, distance, warnings = compute_route(
            lonlat_waypoints[0],
            lonlat_waypoints[1],
            grid,
            bathy,
            land,
            tss,
            tss_vector,
            canals,
            weights,
            draft_m,
            corridor_width,
        )
        # Apply land-guard refinement using vector polygons when available
        try:
            if land is not None:
                shp = Path(__file__).resolve().parents[3] / "data" / "raw" / "land_polygons.shp"
                if shp.exists():
                    land_index = build_land_index(shp)

                    # Global router wrapper returning the already-computed path
                    class _GlobalRouter:
                        def __init__(self, p):
                            self._p = p
                        def route(self, s, e):
                            return self._p

                    # Local router factory used by land guard for refinements
                    def _local_router_factory(local_grid, local_land_mask):
                        import numpy as _np

                        # Use a deep synthetic bathy to allow local re-routing around land
                        bathy_local = _np.full((local_grid.height, local_grid.width), -10000, dtype=_np.int16)

                        class _LocalBathy:
                            def __init__(self, depth):
                                self.depth = depth
                                self.nodata = -32768
                            def is_safe(self, y, x, min_draft):
                                return True
                            def depth_penalty(self, y, x, min_draft, near_threshold_penalty=0.0):
                                return 0.0

                        local_astar = CoarseToFineAStar(
                            local_grid,
                            bathy_local,
                            land_mask=local_land_mask,
                            coarse_scale=8,
                        )

                        class _LocalProvider:
                            def route(self, a, b):
                                ctx = CostContext(
                                    bathy=_LocalBathy(bathy_local),
                                    tss=None,
                                    density=None,
                                    grid_dx=local_grid.dx,
                                    grid_dy=local_grid.dy,
                                    goal_bearing=0.0,
                                    land=None,
                                    grid=local_grid,
                                    tss_wrong_way_hard=False,
                                    tss_snap_lane_graph=False,
                                    tss_disable_lane_smoothing=False,
                                )
                                w = CostWeights()
                                res = local_astar.search(a, b, ctx, w, min_draft=1.0)
                                return res.path if res.success else []

                        return _LocalProvider()

                    params = LandGuardParams(local_router_factory=_local_router_factory)
                    global_wrapper = _GlobalRouter(path)
                    guarded = route_with_land_guard(lonlat_waypoints[0], lonlat_waypoints[1], global_wrapper, land_index, params)
                    path = guarded
        except Exception as e:
            warnings.append(f"Land guard error: {e}; continuing with original path")
    else:
        stitched_path: List[Tuple[float, float]] = []
        for idx in range(len(lonlat_waypoints) - 1):
            segment_path, _, segment_warnings = compute_route(
                lonlat_waypoints[idx],
                lonlat_waypoints[idx + 1],
                grid,
                bathy,
                land,
                tss,
                tss_vector,
                canals,
                weights,
                draft_m,
                corridor_width,
            )
            warnings.extend(segment_warnings)
            if not stitched_path:
                stitched_path.extend(segment_path)
            else:
                stitched_path.extend(segment_path[1:])

        min_land_dist_cells = max(2, int(cfg.land.min_distance_nm / (grid.dx * 60)))
        overall_dist = rhumb_distance_nm(
            lonlat_waypoints[0][0],
            lonlat_waypoints[0][1],
            lonlat_waypoints[-1][0],
            lonlat_waypoints[-1][1],
        )
        adaptive_max_simplify = max(cfg.simplify.max_simplify_nm, min(1000.0, overall_dist * 0.5))
        path, _ = tss_aware_simplify(
            stitched_path,
            grid,
            tss,
            land,
            preserve_points=lonlat_waypoints,
            max_simplify_nm=adaptive_max_simplify,
            min_land_distance_cells=min_land_dist_cells,
            snap_to_lane_graph=tss_snap_lane_graph,
            disable_lane_smoothing=tss_disable_lane_smoothing,
        )

        # For multi-segment requests, run a land-guard refinement on the final stitched path
        try:
            if land is not None:
                shp = Path(__file__).resolve().parents[3] / "data" / "raw" / "land_polygons.shp"
                if shp.exists():
                    land_index = build_land_index(shp)

                    class _GlobalRouter2:
                        def __init__(self, p):
                            self._p = p
                        def route(self, s, e):
                            return self._p

                    def _local_router_factory2(local_grid, local_land_mask):
                        import numpy as _np

                        bathy_local = _np.full((local_grid.height, local_grid.width), -10000, dtype=_np.int16)

                        class _LocalBathy2:
                            def __init__(self, depth):
                                self.depth = depth
                                self.nodata = -32768
                            def is_safe(self, y, x, min_draft):
                                return True
                            def depth_penalty(self, y, x, min_draft, near_threshold_penalty=0.0):
                                return 0.0

                        local_astar = CoarseToFineAStar(local_grid, bathy_local, land_mask=local_land_mask, coarse_scale=8)

                        class _LocalProvider2:
                            def route(self, a, b):
                                ctx = CostContext(
                                    bathy=_LocalBathy2(bathy_local),
                                    tss=None,
                                    density=None,
                                    grid_dx=local_grid.dx,
                                    grid_dy=local_grid.dy,
                                    goal_bearing=0.0,
                                    land=None,
                                    grid=local_grid,
                                    tss_wrong_way_hard=False,
                                    tss_snap_lane_graph=False,
                                    tss_disable_lane_smoothing=False,
                                )
                                w = CostWeights()
                                res = local_astar.search(a, b, ctx, w, min_draft=1.0)
                                return res.path if res.success else []

                        return _LocalProvider2()

                    params2 = LandGuardParams(local_router_factory=_local_router_factory2)
                    global_wrapper2 = _GlobalRouter2(path)
                    guarded2 = route_with_land_guard(lonlat_waypoints[0], lonlat_waypoints[-1], global_wrapper2, land_index, params2)
                    path = guarded2
        except Exception as e:
            warnings.append(f"Land guard error: {e}; continuing with original path")
        if tss is not None:
            path, num_repairs = repair_tss_violations(
                path,
                grid,
                tss,
                land,
                min_land_dist_cells,
                bypass_offsets_nm=cfg.bypass.offset_distances_nm,
            )
            if num_repairs > 0:
                print(f"[TSS REPAIR] Made {num_repairs} repairs to avoid wrong-way TSS segments")
        distance = 0.0
        for i in range(len(path) - 1):
            distance += rhumb_distance_nm(
                path[i][0],
                path[i][1],
                path[i + 1][0],
                path[i + 1][1],
            )
    
    # Convert output path back to [lat, lon] format
    t0 = time.perf_counter()
    path = [(lat, lon) for lon, lat in path]
    print(f"[TIMING] Path format conversion: {(time.perf_counter() - t0)*1000:.1f}ms")
    
    t_endpoint_total = time.perf_counter() - t_endpoint_start
    print(f"[TIMING] ========== ROUTE REQUEST COMPLETE: {t_endpoint_total*1000:.1f}ms ({t_endpoint_total:.2f}s) ==========\n")
    
    return RouteResponse(
        path=path,
        distance_nm=distance,
        macro_route=None,
        warnings=warnings,
    )
