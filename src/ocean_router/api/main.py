"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocean_router.api.endpoints import router as route_router


app = FastAPI(title="Direct Ocean Router")

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
