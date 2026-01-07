"""Traffic Separation Scheme helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ocean_router.core.geodesy import angle_diff_deg
from ocean_router.core.memmaps import MemMapLoader

if TYPE_CHECKING:
    from ocean_router.core.grid import GridSpec


@dataclass
class TSSFields:
    lane_mask_path: Path
    direction_field_path: Path
    sepzone_mask_path: Optional[Path] = None
    sepboundary_mask_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self._lane_mask = MemMapLoader(self.lane_mask_path, dtype=np.uint8)
        self._dir_field = MemMapLoader(self.direction_field_path, dtype=np.int16)
        self._sepzone_mask = MemMapLoader(self.sepzone_mask_path, dtype=np.uint8) if self.sepzone_mask_path else None
        self._sepboundary_mask = MemMapLoader(self.sepboundary_mask_path, dtype=np.uint8) if self.sepboundary_mask_path else None

    @property
    def lane_mask(self) -> np.memmap:
        return self._lane_mask.array

    @property
    def direction_field(self) -> np.memmap:
        return self._dir_field.array

    @property
    def sepzone_mask(self) -> Optional[np.memmap]:
        return self._sepzone_mask.array if self._sepzone_mask else None

    @property
    def sepboundary_mask(self) -> Optional[np.memmap]:
        return self._sepboundary_mask.array if self._sepboundary_mask else None

    def alignment_penalty(
        self, 
        y: int, 
        x: int, 
        move_bearing: float, 
        wrong_way_penalty: float, 
        alignment_weight: float,
        prev_y: int = None,
        prev_x: int = None,
        goal_bearing: float = None,
        max_lane_deviation_deg: float = 45.0,
    ) -> float:
        """Calculate alignment penalty, checking along the entire path.
        
        If prev_y/prev_x are provided, uses Bresenham to check every cell
        along the path for wrong-way violations (important for narrow lanes).
        
        If goal_bearing is provided, penalizes TSS lanes that deviate too much
        from the overall route direction.
        """
        # If we have a previous position, check alignment along the entire path
        if prev_y is not None and prev_x is not None:
            penalty = self._alignment_along_path(
                prev_y, prev_x, y, x, move_bearing, wrong_way_penalty, alignment_weight, 
                goal_bearing, max_lane_deviation_deg
            )
            return penalty
        
        # Fallback: just check destination cell
        return self._single_cell_alignment(y, x, move_bearing, wrong_way_penalty, alignment_weight,
                                          goal_bearing, max_lane_deviation_deg)
    
    def _single_cell_alignment(
        self, y: int, x: int, move_bearing: float, wrong_way_penalty: float, alignment_weight: float,
        goal_bearing: float = None, max_lane_deviation_deg: float = 100.0
    ) -> float:
        """Check alignment at a single cell."""
        if not bool(self.lane_mask[y, x]):
            return 0.0  # No penalty outside TSS lanes
        preferred = float(self.direction_field[y, x])
        angle = angle_diff_deg(move_bearing, preferred)
        
        # Check if TSS lane direction deviates too much from goal bearing
        if goal_bearing is not None:
            lane_to_goal_angle = angle_diff_deg(preferred, goal_bearing)
            if lane_to_goal_angle > max_lane_deviation_deg:
                # TSS lane goes in wrong overall direction - don't use it
                return wrong_way_penalty
        
        if angle > 80:
            return wrong_way_penalty
        else:
            return -alignment_weight * (1.0 - angle / 90.0)
    
    def _alignment_along_path(
        self,
        y0: int, x0: int,
        y1: int, x1: int,
        move_bearing: float,
        wrong_way_penalty: float,
        alignment_weight: float,
        goal_bearing: float = None,
        max_lane_deviation_deg: float = 25.0,
        proximity_check_radius: int = 2,  # ~1nm at 0.5nm resolution
    ) -> float:
        """Check alignment along entire path using Bresenham.
        
        Returns the maximum penalty encountered along the path.
        If any cell is wrong-way, returns the wrong_way_penalty.
        Also checks cells within proximity_check_radius for wrong-way lanes.
        Otherwise returns the best bonus from cells we pass through.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        max_penalty = 0.0
        min_bonus = 0.0
        cells_in_lane = 0
        wrong_way_detected = False
        
        while True:
            # Check bounds
            if 0 <= y < self.lane_mask.shape[0] and 0 <= x < self.lane_mask.shape[1]:
                if bool(self.lane_mask[y, x]):
                    # We're directly in a lane
                    cells_in_lane += 1
                    preferred = float(self.direction_field[y, x])
                    
                    # Check if TSS lane direction deviates too much from goal bearing
                    if goal_bearing is not None:
                        lane_to_goal_angle = angle_diff_deg(preferred, goal_bearing)
                        if lane_to_goal_angle > 45:
                            # TSS lane goes in wrong overall direction
                            return wrong_way_penalty * 0.5
                    
                    angle = angle_diff_deg(move_bearing, preferred)
                    
                    if angle > 90:
                        # Wrong way - immediate large penalty
                        return wrong_way_penalty
                    else:
                        # Right way - track best bonus
                        bonus = -alignment_weight * (1.0 - angle / 90.0)
                        min_bonus = min(min_bonus, bonus)
                else:
                    # Not in a lane - check if we're close to a wrong-way lane
                    nearby_wrong_way = self._check_nearby_wrong_way(
                        y, x, move_bearing, proximity_check_radius
                    )
                    if nearby_wrong_way:
                        # Apply partial penalty for being near wrong-way lane
                        # Scale by distance - closer = higher penalty
                        return wrong_way_penalty * 0.5
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Return the bonus if we passed through any lanes correctly
        if cells_in_lane > 0:
            return min_bonus
        return 0.0

    def _check_nearby_wrong_way(
        self, y: int, x: int, move_bearing: float, radius: int = 2
    ) -> bool:
        """Check if there's a wrong-way TSS lane within radius cells.
        
        Only returns True if a nearby lane is going the wrong way AND
        there's no right-way lane closer or at the same distance.
        
        Args:
            y, x: Current cell coordinates
            move_bearing: Direction we're traveling
            radius: Search radius in cells (~1nm at 0.5nm resolution = 2 cells)
            
        Returns:
            True if there's a wrong-way lane nearby without a right-way alternative
        """
        height, width = self.lane_mask.shape
        
        closest_wrong_way_dist = float('inf')
        closest_right_way_dist = float('inf')
        
        # Check all cells within radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue  # Skip center cell (already checked)
                    
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                
                if not bool(self.lane_mask[ny, nx]):
                    continue  # Not a lane
                
                # Calculate distance
                dist = (dy * dy + dx * dx) ** 0.5
                if dist > radius:
                    continue  # Outside radius
                
                # Check direction
                preferred = float(self.direction_field[ny, nx])
                angle = angle_diff_deg(move_bearing, preferred)
                
                if angle > 90:
                    # Wrong way lane nearby
                    closest_wrong_way_dist = min(closest_wrong_way_dist, dist)
                else:
                    # Right way lane nearby
                    closest_right_way_dist = min(closest_right_way_dist, dist)
        
        # Only penalize if wrong-way lane is closer than any right-way lane
        # This allows routes to pass through right-way lanes even if wrong-way is nearby
        if closest_wrong_way_dist < float('inf'):
            if closest_right_way_dist >= closest_wrong_way_dist:
                return True  # Wrong-way is closer or same distance
        
        return False

    def in_or_near_lane(self, y: int, x: int, radius: int = 2) -> bool:
        """Fast check if cell is in or near a TSS lane (within radius cells).
        
        Used to avoid expensive calculations when nowhere near a TSS.
        """
        if bool(self.lane_mask[y, x]):
            return True
        
        # Quick check of immediate neighbors only
        height, width = self.lane_mask.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if bool(self.lane_mask[ny, nx]):
                        return True
        return False

    def in_lane(self, y: int, x: int) -> bool:
        """Check if cell is within a TSS lane."""
        return bool(self.lane_mask[y, x])

    def in_sepzone(self, y: int, x: int) -> bool:
        """Check if cell is within a separation zone."""
        if self.sepzone_mask is None:
            return False
        return bool(self.sepzone_mask[y, x])

    def in_sepboundary(self, y: int, x: int) -> bool:
        """Check if cell is on a separation boundary line."""
        if self.sepboundary_mask is None:
            return False
        return bool(self.sepboundary_mask[y, x])

    def boundary_crossing_penalty(
        self,
        prev_y: int,
        prev_x: int,
        y: int,
        x: int,
        lane_crossing_penalty: float,
        sepzone_crossing_penalty: float,
        sepboundary_crossing_penalty: float = 0.0,
    ) -> float:
        """Calculate penalty for crossing TSS boundaries.
        
        Penalizes:
        - Exiting a TSS lane (encourages staying in lanes once entered)
        - Entering a separation zone (should be avoided entirely)
        - Crossing a separation boundary line
        
        Args:
            prev_y, prev_x: Previous cell coordinates
            y, x: Current cell coordinates
            lane_crossing_penalty: Penalty for EXITING lane boundaries (not entering)
            sepzone_crossing_penalty: Penalty for entering separation zones
            sepboundary_crossing_penalty: Penalty for crossing separation boundary lines
            
        Returns:
            Total crossing penalty
        """
        penalty = 0.0
        
        # Check lane boundary crossing - only penalize EXITING lanes
        # (entering should be encouraged, staying in lane is rewarded by alignment bonus)
        was_in_lane = self.in_lane(prev_y, prev_x)
        now_in_lane = self.in_lane(y, x)
        if was_in_lane and not now_in_lane:
            # Exiting a lane - mild penalty to encourage staying in lanes
            penalty += lane_crossing_penalty
        
        # Check separation zone crossing (entering is heavily penalized)
        was_in_sepzone = self.in_sepzone(prev_y, prev_x)
        now_in_sepzone = self.in_sepzone(y, x)
        if not was_in_sepzone and now_in_sepzone:
            # Entering a separation zone - heavy penalty
            penalty += sepzone_crossing_penalty
        elif was_in_sepzone and not now_in_sepzone:
            # Exiting a separation zone - smaller penalty to encourage leaving
            penalty += sepzone_crossing_penalty * 0.25

        if self.line_crosses_boundary(prev_y, prev_x, y, x):
            penalty += sepzone_crossing_penalty
        
        # Check separation boundary crossing (crossing the line itself)
        # Always check if we crossed over a boundary line using Bresenham
        # This catches both landing on boundaries AND stepping over them
        if self._line_crosses_sepboundary(prev_y, prev_x, y, x):
            penalty += sepboundary_crossing_penalty
        
        return penalty

    def _line_crosses_sepboundary(self, y0: int, x0: int, y1: int, x1: int) -> bool:
        """Check if a line crosses a separation boundary without landing on it."""
        if self.sepboundary_mask is None:
            return False
            
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            
            # Check bounds and if we hit a boundary
            if 0 <= y < self.lane_mask.shape[0] and 0 <= x < self.lane_mask.shape[1]:
                if bool(self.sepboundary_mask[y, x]):
                    return True
        
        return False

    def line_crosses_boundary(self, y0: int, x0: int, y1: int, x1: int) -> bool:
        """Check if a line between two points crosses critical TSS boundaries.
        
        Uses Bresenham's line algorithm to walk along the path and detect
        crossings of separation zones or separation boundary lines.
        
        Note: Lane-to-lane transitions are allowed (e.g., staying in TSS but moving
        between lanes is OK). Only separation zone/boundary crossings block smoothing.
        The wrong-way check is done separately in line_goes_wrong_way().
        
        Args:
            y0, x0: Start cell coordinates
            y1, x1: End cell coordinates
            
        Returns:
            True if the line crosses a separation zone boundary or separation line
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        prev_in_sepzone = self.in_sepzone(y, x)
        prev_on_sepboundary = self.in_sepboundary(y, x)
        
        while True:
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            
            # Check bounds
            if not (0 <= y < self.lane_mask.shape[0] and 0 <= x < self.lane_mask.shape[1]):
                continue
            
            # Check if we crossed a separation zone boundary (entering OR exiting)
            curr_in_sepzone = self.in_sepzone(y, x)
            if curr_in_sepzone != prev_in_sepzone:
                return True  # Crossed separation zone boundary
            prev_in_sepzone = curr_in_sepzone
            
            # Check if we crossed a separation boundary line
            curr_on_sepboundary = self.in_sepboundary(y, x)
            if not prev_on_sepboundary and curr_on_sepboundary:
                return True  # Crossed separation boundary line
            prev_on_sepboundary = curr_on_sepboundary
        
        return False

    def line_goes_wrong_way(self, y0: int, x0: int, y1: int, x1: int, grid: Optional['GridSpec'] = None) -> bool:
        """Check if a line between two points goes against traffic in any TSS lane.
        
        Uses Bresenham's line algorithm to walk along the path and check if any
        cell along the line is in a TSS lane going the wrong direction.
        
        Args:
            y0, x0: Start cell coordinates
            y1, x1: End cell coordinates
            grid: Optional GridSpec for accurate bearing calculation
            
        Returns:
            True if any cell along the line is wrong-way (>90° from preferred)
        """
        import math
        
        # Calculate overall bearing of the line
        dx_total = x1 - x0
        dy_total = y1 - y0
        if dx_total == 0 and dy_total == 0:
            return False
        
        # Use geodetic bearing if grid is provided (more accurate)
        if grid is not None:
            lon0, lat0 = grid.xy_to_lonlat(x0, y0)
            lon1, lat1 = grid.xy_to_lonlat(x1, y1)
            from ocean_router.core.geodesy import bearing_deg
            move_bearing = bearing_deg(lon0, lat0, lon1, lat1)
        else:
            # Fallback: grid-based bearing with latitude correction
            # Approximate latitude for scaling (use middle of line)
            mid_y = (y0 + y1) / 2
            # Approximate latitude: assumes standard grid where y=0 is at some lat_max
            # This is a rough correction - ideally use grid.xy_to_lonlat
            approx_lat = 90 - (mid_y * 0.0083333333)  # Rough estimate for 0.5nm grid
            lat_scale = math.cos(math.radians(approx_lat)) if abs(approx_lat) < 89 else 0.1
            dx_scaled = dx_total * lat_scale
            move_bearing = math.degrees(math.atan2(dx_scaled, -dy_total)) % 360
        
        # Walk the line using Bresenham
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Check bounds
            if 0 <= y < self.lane_mask.shape[0] and 0 <= x < self.lane_mask.shape[1]:
                if bool(self.lane_mask[y, x]):
                    preferred = float(self.direction_field[y, x])
                    angle = angle_diff_deg(move_bearing, preferred)
                    if angle > 90:
                        return True  # Wrong way detected
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return False

    def blocked(self, y: int, x: int) -> bool:
        if self.sepzone_mask is None:
            return False
        return bool(self.sepzone_mask[y, x])
