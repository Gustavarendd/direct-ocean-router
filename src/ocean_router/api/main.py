"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from ocean_router.api.endpoints import router as route_router


app = FastAPI(title="Direct Ocean Router")
app.include_router(route_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
