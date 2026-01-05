"""API request and response models."""
from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    start: Tuple[float, float] = Field(..., description="(lon, lat) of start point")
    end: Tuple[float, float] = Field(..., description="(lon, lat) of end point")
    draft_m: float = Field(..., description="Vessel draft in meters")
    safety_margin_m: float = Field(2.0, description="Additional safety margin in meters")
    corridor_width_nm: float = Field(500.0, description="Offshore buffer for corridor")


class RouteResponse(BaseModel):
    path: List[Tuple[float, float]]
    distance_nm: float
    macro_route: List[str] | None = None
    warnings: List[str] = []
