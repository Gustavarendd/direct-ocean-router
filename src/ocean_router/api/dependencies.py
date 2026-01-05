"""Dependency wiring for API service."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from ocean_router.core.grid import GridSpec
from ocean_router.data.bathy import Bathy, load_bathy
from ocean_router.data.land import LandMask
from ocean_router.data.tss import TSSFields


def _project_root() -> Path:
    """Get project root (4 levels up from this file)."""
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def get_grid_spec() -> GridSpec:
    cfg_path = _project_root() / "configs" / "grid_1nm.json"
    return GridSpec.from_file(cfg_path)


@lru_cache(maxsize=1)
def get_bathy() -> Optional[Bathy]:
    path = _project_root() / "data" / "processed" / "bathy" / "depth_1nm.npy"
    if not path.exists():
        return None
    return load_bathy(path)


@lru_cache(maxsize=1)
def get_land_mask() -> Optional[LandMask]:
    base_path = _project_root() / "data" / "processed" / "land" / "land_mask_1nm.npy"
    buffered_path = _project_root() / "data" / "processed" / "land" / "land_mask_1nm_buffered.npy"
    if not base_path.exists():
        return None
    buff = buffered_path if buffered_path.exists() else None
    return LandMask(base_path, buff)


@lru_cache(maxsize=1)
def get_tss() -> Optional[TSSFields]:
    tss_dir = _project_root() / "data" / "processed" / "tss"
    lane_mask = tss_dir / "tss_lane_mask_1nm.npy"
    dir_field = tss_dir / "tss_dir_field_1nm.npy"
    sep_mask = tss_dir / "tss_sepzone_mask_1nm.npy"
    if not lane_mask.exists() or not dir_field.exists():
        return None
    return TSSFields(
        lane_mask_path=lane_mask,
        direction_field_path=dir_field,
        sepzone_mask_path=sep_mask if sep_mask.exists() else None,
    )
