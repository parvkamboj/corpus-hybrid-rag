from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings

_bearer = HTTPBearer()


async def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db.session_factory() as session:
        yield session


async def get_redis(request: Request) -> Redis:  # type: ignore[type-arg]
    return request.app.state.redis  # type: ignore[no-any-return]


async def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant  # type: ignore[no-any-return]
