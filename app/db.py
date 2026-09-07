"""
Database engine, session factory, and startup initialisation.

Schema creation is done with create_all at startup rather than with Alembic.
That is the right trade while the schema is still moving; the moment the
dashboard or the eval harness depends on a stable shape, this should become a
migration directory.

Everything that touches the network here is bounded by a timeout. The service
is designed to run without a database — reporting that fact rather than
failing — and an unbounded connect attempt breaks that promise in the worst
way: the process starts, binds its port, and then never answers anything.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.models import Base

log = logging.getLogger(__name__)

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=False,
    pool_pre_ping=True,  # survives Postgres restarts during a demo
    # Bounds the TCP connect. Without it, a host that silently drops packets
    # (a stopped container, a firewalled port) hangs the caller indefinitely
    # instead of raising something the caller can handle.
    connect_args={"connect_timeout": int(_settings.db_connect_timeout_seconds)},
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """
    Enable pgvector and create any missing tables. Safe to run repeatedly.

    Raises on failure — including TimeoutError — so the caller decides whether
    a missing database is fatal. app.main treats it as a degraded start.
    """

    async def _run() -> None:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

    await asyncio.wait_for(_run(), timeout=_settings.db_startup_timeout_seconds)
    log.info("database ready (pgvector enabled, %d tables)", len(Base.metadata.tables))


async def ping() -> bool:
    """True if the database answers within the connect timeout. Used by /health."""
    try:
        await asyncio.wait_for(
            _select_one(), timeout=_settings.db_connect_timeout_seconds + 1.0
        )
        return True
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        log.debug("database ping failed: %s", exc)
        return False


async def _select_one() -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency yielding a session that is committed or rolled back.

    The commit is conditional on a transaction actually being open. A route
    that already handled its own persistence failure will have rolled back, and
    committing again would raise a second, more confusing error from inside
    dependency teardown — after the response body has been built.
    """
    async with SessionLocal() as session:
        try:
            yield session
            if session.in_transaction():
                await session.commit()
        except Exception:
            await session.rollback()
            raise
