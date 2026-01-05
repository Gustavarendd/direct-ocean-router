"""Cost components for corridor-limited A*."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ocean_router.core.geodesy import angle_diff_deg, move_cost_nm
from ocean_router.data.bathy import Bathy
from ocean_router.data.tss import TSSFields
from ocean_router.data.density import Density


@dataclass
class CostWeights:
    tss_wrong_way_penalty: float = 50.0
    tss_alignment_weight: float = 10.0
    near_shore_depth_penalty: float = 5.0
    density_bias: float = -2.0
    turn_penalty_weight: float = 0.5


@dataclass
class CostContext:
    bathy: Bathy
    tss: Optional[TSSFields]
    density: Optional[Density]
    grid_dx: float
    grid_dy: float

    def move_cost(self, y: int, x: int, lat: float, prev_bearing: Optional[float], move_bearing: float, min_draft: float, weights: CostWeights) -> float:
        """Calculate cost to move to cell (y, x).
        
        Args:
            min_draft: Minimum required water depth in meters (positive value)
        """
        step_cost = move_cost_nm(self.grid_dx, self.grid_dy, lat)
        penalty = 0.0
        penalty += self.bathy.depth_penalty(y, x, min_draft, weights.near_shore_depth_penalty)
        if self.tss:
            penalty += self.tss.alignment_penalty(y, x, move_bearing, weights.tss_wrong_way_penalty, weights.tss_alignment_weight)
        if self.density:
            penalty += self.density.bias(y, x, weights.density_bias)
        if prev_bearing is not None:
            penalty += weights.turn_penalty_weight * angle_diff_deg(prev_bearing, move_bearing) / 180
        return step_cost + penalty

    def blocked(self, y: int, x: int, min_draft: float) -> bool:
        """Check if cell is blocked (too shallow or in separation zone).
        
        Args:
            min_draft: Minimum required water depth in meters (positive value)
        """
        if not self.bathy.is_safe(y, x, min_draft):
            return True
        if self.tss and self.tss.blocked(y, x):
            return True
        return False
