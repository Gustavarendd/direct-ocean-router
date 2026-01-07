"""API routers."""
from __future__ import annotations

from typing import List, Tuple, Optional
from fastapi import APIRouter, Depends, HTTPException

from ocean_router.api.schemas import RouteRequest, RouteResponse
from ocean_router.api.dependencies import get_grid_spec, get_bathy, get_land_mask, get_tss, get_cost_weights
from ocean_router.core.geodesy import haversine_nm
from ocean_router.core.grid import GridSpec
from ocean_router.core.config import get_config
from ocean_router.data.bathy import Bathy
from ocean_router.data.land import LandMask
from ocean_router.data.tss import TSSFields
from ocean_router.routing.costs import CostContext, CostWeights
from ocean_router.routing.corridor import build_corridor
from ocean_router.routing.astar import CorridorAStar
from ocean_router.routing.astar_fast import CoarseToFineAStar, FastCorridorAStar
from ocean_router.routing.simplify import simplify_path, simplify_between_tss_boundaries, tss_aware_simplify, repair_tss_violations

# Cache for CoarseToFineAStar (expensive to initialize)
# We keep two caches - one for short routes (wider corridor) and one for long routes
_coarse_to_fine_cache_short: Optional["CoarseToFineAStar"] = None
_coarse_to_fine_cache_long: Optional["CoarseToFineAStar"] = None

router = APIRouter()


def compute_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    grid: GridSpec,
    bathy: Optional[Bathy],
    land: Optional[LandMask],
    tss: Optional[TSSFields],
    weights: CostWeights,
    draft_m: float = 10.0,
    corridor_width_nm: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], float, List[str]]:
    """Compute a route avoiding land and shallow water."""
    global _coarse_to_fine_cache_short, _coarse_to_fine_cache_long
    warnings: List[str] = []
    
    # Load config
    cfg = get_config()
    
    # Use config defaults if not specified
    if corridor_width_nm is None:
        corridor_width_nm = cfg.corridor.width_short_nm
    
    # If no preprocessed data, return great circle
    if bathy is None:
        warnings.append("Bathymetry data not loaded; returning great-circle segment only.")
        distance = haversine_nm(start[0], start[1], end[0], end[1])
        return [start, end], distance, warnings
    
    # Draft is a positive value (e.g., 10m means vessel needs 10m of water)
    min_draft = draft_m
    
    try:
        # Calculate overall bearing from start to end for TSS filtering
        import math
        lon1, lat1 = start
        lon2, lat2 = end
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        goal_bearing = math.degrees(math.atan2(dlon * math.cos(math.radians((lat1 + lat2) / 2)), dlat)) % 360
        
        # Create cost context
        context = CostContext(
            bathy=bathy,
            tss=tss,
            density=None,
            grid_dx=grid.dx,
            grid_dy=grid.dy,
            goal_bearing=goal_bearing,
            land=land,
        )
        print(f"[ROUTE] CostContext created with land={'present' if land else 'None'}")
        
        # Pre-compute distance transform to enable land proximity penalties during search
        if land is not None:
            _ = land.distance_from_land  # Force computation via cached_property access
        
        # Choose algorithm based on distance and config
        straight_line_dist = haversine_nm(start[0], start[1], end[0], end[1])
        
        use_coarse_to_fine = (
            cfg.algorithm.default == "coarse_to_fine" or
            (cfg.algorithm.default == "auto" and 
             straight_line_dist > cfg.algorithm.coarse_to_fine_threshold_nm)
        )
        
        if use_coarse_to_fine:
            # Use coarse-to-fine A* for long-distance routes
            # Use wider corridor for medium routes (may pass through straits like Florida)
            # Use narrower corridor for very long routes (mostly open ocean)
            if straight_line_dist > cfg.corridor.long_route_threshold_nm:
                # Long route - use narrow corridor
                if _coarse_to_fine_cache_long is None:
                    _coarse_to_fine_cache_long = CoarseToFineAStar(
                        grid,
                        bathy.depth,
                        land_mask=land.buffered if land else None,
                        coarse_scale=cfg.algorithm.coarse_scale,
                        corridor_width_nm=cfg.corridor.width_long_nm
                    )
                astar = _coarse_to_fine_cache_long
            else:
                # Medium route - use wider corridor for straits
                if _coarse_to_fine_cache_short is None:
                    _coarse_to_fine_cache_short = CoarseToFineAStar(
                        grid,
                        bathy.depth,
                        land_mask=land.buffered if land else None,
                        coarse_scale=cfg.algorithm.coarse_scale,
                        corridor_width_nm=cfg.corridor.width_short_nm
                    )
                astar = _coarse_to_fine_cache_short
        else:
            # Use corridor-based A* for shorter routes
            corridor_mask, x_off, y_off = build_corridor(
                grid, start, end, 
                land_mask=land.buffered if land else None,
                width_nm=corridor_width_nm,
            )
            astar = FastCorridorAStar(grid, corridor_mask, x_off, y_off)
        
        result = astar.search(start, end, context, weights, min_draft)
        
        print(f"[ROUTE RESULT] explored={result.explored}, cost={result.cost:.1f}nm, path_len={len(result.path)}")
        
        # Check final path for TSS violations
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
        
        if not result.success:
            warnings.append("A* search did not find a valid path; returning great-circle.")
            distance = haversine_nm(start[0], start[1], end[0], end[1])
            return [start, end], distance, warnings
        
        # TSS-aware path simplification:
        # 1. Collapse TSS-to-TSS spans if a straight water path exists
        # 2. Try to skip intermediate TSS runs (A -> B -> C => go A -> C if clear)
        # 3. In non-TSS stretches, greedily remove intermediates when straight path is clear
        # Convert minimum land distance to cells based on grid resolution
        min_land_dist_cells = max(2, int(cfg.land.min_distance_nm / (grid.dx * 60)))
        path, in_tss = tss_aware_simplify(
            result.path, 
            grid, 
            tss, 
            land,
            preserve_points=None,
            max_simplify_nm=cfg.simplify.max_simplify_nm,
            min_land_distance_cells=min_land_dist_cells
        )
        
        print(f"[SIMPLIFY] {len(result.path)} -> {len(path)} waypoints")
        
        # Repair any TSS violations in the path (wrong-way segments)
        if tss is not None:
            path, num_repairs = repair_tss_violations(
                path, grid, tss, land, min_land_dist_cells,
                bypass_offsets_nm=cfg.bypass.offset_distances_nm
            )
            if num_repairs > 0:
                print(f"[TSS REPAIR] Made {num_repairs} repairs to avoid wrong-way TSS segments")
        
        # Calculate total distance
        total_dist = 0.0
        for i in range(len(path) - 1):
            total_dist += haversine_nm(
                path[i][0], path[i][1],
                path[i+1][0], path[i+1][1]
            )
        
        return path, total_dist, warnings
        
    except Exception as e:
        warnings.append(f"Routing error: {str(e)}; returning great-circle segment.")
        distance = haversine_nm(start[0], start[1], end[0], end[1])
        return [start, end], distance, warnings


@router.post("/route", response_model=RouteResponse)
def route(
    req: RouteRequest,
    grid: GridSpec = Depends(get_grid_spec),
    bathy: Optional[Bathy] = Depends(get_bathy),
    land: Optional[LandMask] = Depends(get_land_mask),
    tss: Optional[TSSFields] = Depends(get_tss),
    weights: CostWeights = Depends(get_cost_weights),
) -> RouteResponse:
    """Compute a ship route between two points.
    
    Coordinates are expected as [lat, lon] format.
    """
    cfg = get_config()
    
    # Convert input from [lat, lon] to internal [lon, lat] format
    start = (req.start[1], req.start[0])  # (lon, lat)
    end = (req.end[1], req.end[0])  # (lon, lat)
    draft_m = req.draft_m if req.draft_m else 10.0
    corridor_width = req.corridor_width_nm if req.corridor_width_nm else cfg.corridor.width_short_nm
    
    path, distance, warnings = compute_route(
        start, end, grid, bathy, land, tss, weights, draft_m, corridor_width
    )
    
    # Convert output path back to [lat, lon] format
    path = [(lat, lon) for lon, lat in path]
    
    return RouteResponse(
        path=path,
        distance_nm=distance,
        macro_route=None,
        warnings=warnings,
    )
