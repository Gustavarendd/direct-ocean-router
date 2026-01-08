"""Cost components for corridor-limited A*."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ocean_router.core.geodesy import angle_diff_deg, move_cost_nm
from ocean_router.data.bathy import Bathy
from ocean_router.data.tss import TSSFields
from ocean_router.data.density import Density
from ocean_router.data.land import LandMask


@dataclass
class CostWeights:
    tss_wrong_way_penalty: float = 10000.0  # Going wrong way in lane - dangerous
    tss_alignment_weight: float = 1.5   # Modest preference for using TSS lanes correctly
    tss_off_lane_penalty: float = 1.0
    tss_lane_crossing_penalty: float = 5.0  # Small penalty for exiting lanes
    tss_sepzone_crossing_penalty: float = 10000.0  # High penalty for separation zones
    tss_sepboundary_crossing_penalty: float = 5000.0  # Very high penalty for crossing boundary lines
    tss_proximity_check_radius: int = 2
    tss_max_lane_deviation_deg: float = 45.0
    tss_snap_corridor_enabled: bool = True
    tss_snap_corridor_radius_nm: float = 3.0
    near_shore_depth_penalty: float = 5.0
    land_proximity_penalty: float = 100.0  # Strong penalty for being close to land
    land_proximity_max_distance_cells: int = 50  # Max distance to check for land proximity
    density_bias: float = -2.0
    turn_penalty_weight: float = 0.5

    @classmethod
    def from_config(cls, config: dict) -> "CostWeights":
        """Create CostWeights from a config dictionary."""
        weights = cls(
            tss_wrong_way_penalty=config.get("tss_wrong_way_penalty", 10000.0),
            tss_alignment_weight=config.get("tss_alignment_weight", 1.5),
            tss_off_lane_penalty=config.get("tss_off_lane_penalty", 1.0),
            tss_lane_crossing_penalty=config.get("tss_lane_crossing_penalty", 5.0),
            tss_sepzone_crossing_penalty=config.get("tss_sepzone_crossing_penalty", 10000.0),
            tss_sepboundary_crossing_penalty=config.get("tss_sepboundary_crossing_penalty", 5000.0),
            tss_proximity_check_radius=config.get("tss_proximity_check_radius", 2),
            tss_max_lane_deviation_deg=config.get("tss_max_lane_deviation_deg", 45.0),
            tss_snap_corridor_enabled=config.get("tss_snap_corridor_enabled", True),
            tss_snap_corridor_radius_nm=config.get("tss_snap_corridor_radius_nm", 3.0),
            near_shore_depth_penalty=config.get("near_shore_depth_penalty", 5.0),
            land_proximity_penalty=config.get("land_proximity_penalty", 100.0),
            density_bias=config.get("density_bias", -2.0),
            turn_penalty_weight=config.get("turn_penalty_weight", 0.5),
        )
        print(f"[CostWeights] Loaded: wrong_way={weights.tss_wrong_way_penalty}, alignment={weights.tss_alignment_weight}, land_prox={weights.land_proximity_penalty}, sepboundary={weights.tss_sepboundary_crossing_penalty}")
        return weights


@dataclass
class CostContext:
    bathy: Bathy
    tss: Optional[TSSFields]
    density: Optional[Density]
    grid_dx: float
    grid_dy: float
    goal_bearing: Optional[float] = None  # Overall bearing to goal for TSS filtering
    land: Optional[LandMask] = None  # Land mask for proximity penalties

    def move_cost(
        self,
        y: int,
        x: int,
        lat: float,
        prev_bearing: Optional[float],
        move_bearing: float,
        min_draft: float,
        weights: CostWeights,
        prev_y: Optional[int] = None,
        prev_x: Optional[int] = None,
        goal_bearing: Optional[float] = None,
    ) -> float:
        """Calculate cost to move to cell (y, x).
        
        Args:
            min_draft: Minimum required water depth in meters (positive value)
            prev_y: Previous cell y coordinate (for boundary crossing detection)
            prev_x: Previous cell x coordinate (for boundary crossing detection)
            goal_bearing: Optional bearing from the current step toward the goal.
        """
        step_cost = move_cost_nm(self.grid_dx, self.grid_dy, lat)
        penalty = 0.0
        penalty += self.bathy.depth_penalty(y, x, min_draft, weights.near_shore_depth_penalty)
        if self.land and weights.land_proximity_penalty > 0:
            penalty += self.land.proximity_penalty(y, x, max_distance_cells=12, penalty_weight=weights.land_proximity_penalty)
        if self.tss:
            if weights.tss_off_lane_penalty > 0:
                in_near = self.tss.in_or_near_lane(
                    y, x, radius=weights.tss_proximity_check_radius
                )
                if in_near and not self.tss.in_lane(y, x):
                    wrong_only = self.tss._check_nearby_wrong_way(
                        y, x, move_bearing, radius=weights.tss_proximity_check_radius
                    )
                    if not wrong_only:
                        penalty += weights.tss_off_lane_penalty
            penalty += self.tss.alignment_penalty(
                y, x, move_bearing, 
                weights.tss_wrong_way_penalty, 
                weights.tss_alignment_weight,
                prev_y=prev_y, 
                prev_x=prev_x,
                goal_bearing=goal_bearing if goal_bearing is not None else self.goal_bearing,
                max_lane_deviation_deg=weights.tss_max_lane_deviation_deg,
                proximity_check_radius=weights.tss_proximity_check_radius,
            )
            # Add boundary crossing penalties
            if prev_y is not None and prev_x is not None:
                penalty += self.tss.boundary_crossing_penalty(
                    prev_y, prev_x, y, x,
                    weights.tss_lane_crossing_penalty,
                    weights.tss_sepzone_crossing_penalty,
                    weights.tss_sepboundary_crossing_penalty
                )
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
