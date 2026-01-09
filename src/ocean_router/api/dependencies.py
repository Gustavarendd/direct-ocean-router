"""Dependency wiring for API service."""
from __future__ import annotations

import os
import yaml
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from ocean_router.core.grid import GridSpec
from ocean_router.core.config import get_config
from ocean_router.data.bathy import Bathy, load_bathy
from ocean_router.data.canals import Canal, load_canals
from ocean_router.data.land import LandMask
from ocean_router.data.tss import TSSFields
from ocean_router.data.tss_vector import TSSVectorGraph, load_or_build_tss_vector_graph
from ocean_router.routing.costs import CostWeights


# Resolution switching: set OCEAN_ROUTER_RESOLUTION env var to "1nm", "0.5nm", or "0.25nm"
# Example: OCEAN_ROUTER_RESOLUTION=0.5nm python -m ocean_router.cli.main route ...
VALID_RESOLUTIONS = {"1nm", "0.5nm", "0.25nm"}
_resolution_cache: Optional[str] = None


def get_resolution() -> str:
    """Get current resolution from env var or default to 1nm."""
    global _resolution_cache
    if _resolution_cache is None:
        res = os.environ.get("OCEAN_ROUTER_RESOLUTION", "1nm")
        if res not in VALID_RESOLUTIONS:
            print(f"[WARNING] Invalid resolution '{res}', using '1nm'. Valid: {VALID_RESOLUTIONS}")
            res = "1nm"
        
        # Check if requested resolution data exists, otherwise fall back to 1nm
        if res != "1nm":
            suffix = res.replace(".", "")
            data_dir = _project_root() / "data" / "processed"
            tss_check = data_dir / "tss" / f"tss_lane_mask_{suffix}.npy"
            if not tss_check.exists():
                print(f"[WARNING] Data for {res} not found, falling back to 1nm")
                print(f"[WARNING] Build {res} data with: ./scripts/build_all_resolutions.sh {res}")
                res = "1nm"
        
        _resolution_cache = res
        print(f"[RESOLUTION] Using {res} grid")
    return _resolution_cache


def set_resolution(resolution: str) -> None:
    """Programmatically set resolution (clears caches)."""
    global _resolution_cache
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(f"Invalid resolution '{resolution}'. Valid: {VALID_RESOLUTIONS}")
    _resolution_cache = resolution
    os.environ["OCEAN_ROUTER_RESOLUTION"] = resolution
    # Clear cached loaders
    get_grid_spec.cache_clear()
    get_bathy.cache_clear()
    get_land_mask.cache_clear()
    get_tss.cache_clear()
    print(f"[RESOLUTION] Switched to {resolution} grid")


def _project_root() -> Path:
    """Get project root (4 levels up from this file)."""
    return Path(__file__).resolve().parents[3]


def _data_suffix() -> str:
    """Get file suffix based on resolution (e.g., '1nm' or '0.5nm')."""
    return get_resolution().replace(".", "")  # "0.5nm" -> "05nm"


@lru_cache(maxsize=1)
def get_grid_spec() -> GridSpec:
    res = get_resolution()
    cfg_path = _project_root() / "configs" / f"grid_{res}.json"
    if not cfg_path.exists():
        # Fallback to 1nm if requested resolution not available
        print(f"[WARNING] Grid config {cfg_path} not found, falling back to 1nm")
        cfg_path = _project_root() / "configs" / "grid_1nm.json"
    return GridSpec.from_file(cfg_path)


@lru_cache(maxsize=1)
def get_bathy() -> Optional[Bathy]:
    suffix = _data_suffix()
    path = _project_root() / "data" / "processed" / "bathy" / f"depth_{suffix}.npy"
    if not path.exists():
        return None
    return load_bathy(path)


@lru_cache(maxsize=1)
def get_land_mask() -> Optional[LandMask]:
    suffix = _data_suffix()
    base_path = _project_root() / "data" / "processed" / "land" / f"land_mask_{suffix}.npy"
    buffered_path = _project_root() / "data" / "processed" / "land" / f"land_mask_{suffix}_buffered.npy"
    
    if not base_path.exists():
        return None
    
    buff = buffered_path if buffered_path.exists() else None
    return LandMask(base_path, buff)


@lru_cache(maxsize=1)
def get_tss() -> Optional[TSSFields]:
    suffix = _data_suffix()
    tss_dir = _project_root() / "data" / "processed" / "tss"
    lane_mask = tss_dir / f"tss_lane_mask_{suffix}.npy"
    dir_field = tss_dir / f"tss_dir_field_{suffix}.npy"
    sep_mask = tss_dir / f"tss_sepzone_mask_{suffix}.npy"
    sep_boundary = tss_dir / f"tss_sepboundary_mask_{suffix}.npy"
    lane_graph = tss_dir / f"tss_lane_graph_{suffix}.npz"
    
    if not lane_mask.exists() or not dir_field.exists():
        return None
    
    return TSSFields(
        lane_mask_path=lane_mask,
        direction_field_path=dir_field,
        sepzone_mask_path=sep_mask if sep_mask.exists() else None,
        sepboundary_mask_path=sep_boundary if sep_boundary.exists() else None,
        lane_graph_path=lane_graph if lane_graph.exists() else None,
    )


@lru_cache(maxsize=1)
def get_canals() -> List[Canal]:
    """Load canal overrides from config (if present)."""
    path = _project_root() / "configs" / "canals.yaml"
    return load_canals(path)


@lru_cache(maxsize=1)
def get_tss_vector_graph() -> Optional[TSSVectorGraph]:
    cfg = get_config()
    if not cfg.tss.vector_graph_enabled:
        return None
    tss_dir = _project_root() / "data" / "processed" / "tss"
    if cfg.tss.vector_graph_cache_path:
        cache_path = Path(cfg.tss.vector_graph_cache_path)
        if not cache_path.is_absolute():
            cache_path = _project_root() / cache_path
    else:
        cache_path = tss_dir / "tss_vector_graph.pkl"
    geojson_path = None
    if cfg.tss.vector_graph_geojson:
        geojson_path = Path(cfg.tss.vector_graph_geojson)
        if not geojson_path.is_absolute():
            geojson_path = _project_root() / geojson_path
    return load_or_build_tss_vector_graph(
        geojson_path,
        cache_path,
        cfg.tss.vector_graph_sepzone_buffer_nm,
    )


@lru_cache(maxsize=1)
def get_cost_weights() -> CostWeights:
    """Load cost weights from routing_defaults.yaml using config module."""
    cfg = get_config()
    return CostWeights(
        tss_wrong_way_penalty=cfg.tss.wrong_way_penalty,
        tss_alignment_weight=cfg.tss.alignment_weight,
        tss_off_lane_penalty=cfg.tss.off_lane_penalty,
        tss_lane_crossing_penalty=cfg.tss.lane_crossing_penalty,
        tss_sepzone_crossing_penalty=cfg.tss.sepzone_crossing_penalty,
        tss_sepboundary_crossing_penalty=cfg.tss.sepboundary_crossing_penalty,
        tss_proximity_check_radius=cfg.tss.proximity_check_radius,
        tss_max_lane_deviation_deg=cfg.tss.max_lane_deviation_deg,
        tss_snap_corridor_enabled=cfg.tss.snap_corridor_enabled,
        tss_snap_corridor_radius_nm=cfg.tss.snap_corridor_radius_nm,
        tss_lane_graph_lock_enabled=cfg.tss.lane_graph_lock_enabled,
        tss_lane_graph_lock_radius_nm=cfg.tss.lane_graph_lock_radius_nm,
        near_shore_depth_penalty=cfg.depth.near_shore_penalty,
        land_proximity_penalty=cfg.land.proximity_penalty,
        land_proximity_max_distance_cells=cfg.land.max_distance_cells,
    )
