"""
API-key authentication for /analyze.

Two sources of truth, checked in order:
  1. Keys listed in the API_KEYS env var — the bootstrap path, so a fresh
     deployment is usable before anything has been written to the database.
  2. Active rows in the api_keys table, matched on SHA-256 digest.

When neither source has any key configured, auth is disabled and every request
is accepted as "anonymous". That keeps local development frictionless, and
/health reports auth_enabled=false so the state is never silent.

Behaviour when the database is unreachable:

  * Env keys are still enforced. A deployment that configured API_KEYS keeps
    rejecting bad keys even with Postgres down — it fails closed on that path.
  * Database-issued keys cannot be checked, so they are treated as absent.
  * A deployment with no env keys falls back to "anonymous", which is what it
    would have done anyway if the key table were empty.

The gap that leaves is narrow but real: a deployment whose only keys live in
the database, with Postgres down, accepts anonymous callers. That is the
deliberate cost of letting the service run without a database at all. Any
deployment that actually needs enforcement should set API_KEYS as well.
"""

from __future__ import annotations

import hashlib
import logging
import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import ApiKey

log = logging.getLogger(__name__)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A new plaintext key. Shown once by whatever issues it, then discarded."""
    return f"llmsec_{secrets.token_urlsafe(32)}"


async def _lookup(session: AsyncSession, statement):
    """
    Run a key-table query, returning None if the database is unreachable.

    Auth must not be the reason the service 500s when Postgres is down; the
    caller decides what an unavailable key table means.
    """
    try:
        return await session.scalar(statement)
    except Exception as exc:  # noqa: BLE001 - degraded auth, not a crash
        log.warning("api_keys lookup failed, treating as unavailable: %s", exc)
        await session.rollback()
        return None


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> str:
    """
    Returns the label of the authenticated caller, recorded on the analysis row.

    Raises 401 when auth is enabled and the key is missing or unrecognised.
    """
    settings = get_settings()
    env_keys = settings.allowed_api_keys

    if x_api_key and x_api_key in env_keys:
        return "env"

    if x_api_key:
        row = await _lookup(
            session,
            select(ApiKey).where(
                ApiKey.key_hash == hash_key(x_api_key),
                ApiKey.is_active.is_(True),
            ),
        )
        if row is not None:
            return row.label

    # No key matched. Only reject if at least one key exists to match against;
    # a deployment with no keys configured at all is intentionally open.
    has_db_keys = await _lookup(
        session, select(ApiKey.id).where(ApiKey.is_active.is_(True)).limit(1)
    )
    if env_keys or has_db_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return "anonymous"
