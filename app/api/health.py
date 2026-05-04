import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

router = APIRouter()


class ServiceHealth(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    version: str
    services: dict[str, ServiceHealth]


async def _ping_postgres(request: Request) -> ServiceHealth:
    t = time.perf_counter()
    try:
        async with request.app.state.db.session_factory() as session:
            await session.execute(text("SELECT 1"))
        return ServiceHealth(status="ok", latency_ms=round((time.perf_counter() - t) * 1000, 2))
    except Exception as exc:
        return ServiceHealth(status="error", error=str(exc))


async def _ping_redis(request: Request) -> ServiceHealth:
    t = time.perf_counter()
    try:
        await request.app.state.redis.ping()
        return ServiceHealth(status="ok", latency_ms=round((time.perf_counter() - t) * 1000, 2))
    except Exception as exc:
        return ServiceHealth(status="error", error=str(exc))


async def _ping_qdrant(request: Request) -> ServiceHealth:
    t = time.perf_counter()
    try:
        await request.app.state.qdrant.get_collections()
        return ServiceHealth(status="ok", latency_ms=round((time.perf_counter() - t) * 1000, 2))
    except Exception as exc:
        return ServiceHealth(status="error", error=str(exc))


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request, response: Response) -> HealthResponse:
    """Pings Postgres, Redis, and Qdrant. Returns 503 if any service is unreachable."""
    postgres, redis_svc, qdrant = await asyncio.gather(
        _ping_postgres(request),
        _ping_redis(request),
        _ping_qdrant(request),
    )

    services: dict[str, ServiceHealth] = {
        "postgres": postgres,
        "redis": redis_svc,
        "qdrant": qdrant,
    }
    all_ok = all(s.status == "ok" for s in services.values())

    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version="0.1.0",
        services=services,
    )
