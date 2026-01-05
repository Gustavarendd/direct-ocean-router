"""API routers."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ocean_router.api.schemas import RouteRequest, RouteResponse
from ocean_router.api.dependencies import get_grid_spec
from ocean_router.core.geodesy import haversine_nm

router = APIRouter()


@router.post("/route", response_model=RouteResponse)
def route(req: RouteRequest, grid = Depends(get_grid_spec)) -> RouteResponse:  # type: ignore[override]
    distance = haversine_nm(req.start[0], req.start[1], req.end[0], req.end[1])
    warning = "Preprocessed caches not loaded; returning great-circle segment only."
    return RouteResponse(
        path=[req.start, req.end],
        distance_nm=distance,
        macro_route=None,
        warnings=[warning],
    )
