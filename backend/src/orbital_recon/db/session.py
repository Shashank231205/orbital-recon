"""Async database engine and session management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from orbital_recon.core.config import Settings
from orbital_recon.db.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """Create the engine and session factory for the process."""
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    _engine = create_async_engine(
        settings.database_url,
        echo=False,
        # SQLite rejects connections reused across threads by default, and the
        # pipeline runs inference off the event loop in a worker thread.
        connect_args={"check_same_thread": False}
        if settings.database_url.startswith("sqlite")
        else {},
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    return _engine


async def create_schema(engine: AsyncEngine) -> None:
    """Create tables that do not yet exist."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Release pooled connections on shutdown."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database engine is not initialised; call init_engine first")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide a transactional session, rolling back on failure.

    Used by background work that runs outside the request lifecycle and so
    cannot rely on FastAPI dependency injection.
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session."""
    async with session_scope() as session:
        yield session
