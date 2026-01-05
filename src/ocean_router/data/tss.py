"""Traffic Separation Scheme helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ocean_router.core.geodesy import angle_diff_deg
from ocean_router.core.memmaps import MemMapLoader


@dataclass
class TSSFields:
    lane_mask_path: Path
    direction_field_path: Path
    sepzone_mask_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self._lane_mask = MemMapLoader(self.lane_mask_path, dtype=np.uint8)
        self._dir_field = MemMapLoader(self.direction_field_path, dtype=np.int16)
        self._sepzone_mask = MemMapLoader(self.sepzone_mask_path, dtype=np.uint8) if self.sepzone_mask_path else None

    @property
    def lane_mask(self) -> np.memmap:
        return self._lane_mask.array

    @property
    def direction_field(self) -> np.memmap:
        return self._dir_field.array

    @property
    def sepzone_mask(self) -> Optional[np.memmap]:
        return self._sepzone_mask.array if self._sepzone_mask else None

    def alignment_penalty(self, y: int, x: int, move_bearing: float, wrong_way_penalty: float, alignment_weight: float) -> float:
        if not bool(self.lane_mask[y, x]):
            return alignment_weight  # mild encouragement to use lanes
        preferred = float(self.direction_field[y, x])
        angle = angle_diff_deg(move_bearing, preferred)
        penalty = alignment_weight * (angle / 180)
        if angle > 90:
            penalty += wrong_way_penalty
        return penalty

    def blocked(self, y: int, x: int) -> bool:
        if self.sepzone_mask is None:
            return False
        return bool(self.sepzone_mask[y, x])
