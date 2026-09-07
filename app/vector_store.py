"""
pgvector-backed similarity store (PRD Section 12).

One table serves two consumers, partitioned by `kind`:

  "jailbreak_bank"  — the curated bank of known jailbreak prompts. The
                      jailbreak detector embeds an incoming prompt and asks for
                      its nearest neighbours here.
  "context_chunk"   — retrieved RAG chunks, against which the hallucination
                      detector scores each claim for groundedness.

Sharing one table and one ivfflat index means the extension is set up once and
both detectors get the same distance semantics, rather than each rolling its
own similarity code.

Distance is cosine, so `similarity` is 1 - distance and lands in [0, 1] for the
normalised embeddings sentence-transformers produces. The helpers here take
vectors rather than text: embedding is the caller's business, which keeps this
module free of any model dependency and testable without one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Embedding

log = logging.getLogger(__name__)

JAILBREAK_BANK = "jailbreak_bank"
CONTEXT_CHUNK = "context_chunk"


@dataclass(frozen=True)
class Neighbour:
    content: str
    similarity: float
    meta: dict


def _validate(vector: list[float]) -> list[float]:
    """
    Reject wrong-width vectors before they reach Postgres.

    Without this the error surfaces as an opaque driver-level exception mid
    transaction; a mismatch almost always means two different embedding models
    are in play, which is worth saying out loud.
    """
    expected = get_settings().embedding_dim
    if len(vector) != expected:
        raise ValueError(
            f"embedding has {len(vector)} dimensions, expected {expected}. "
            "The store and the embedding model must agree - check EMBEDDING_DIM."
        )
    return vector


async def add(
    session: AsyncSession,
    kind: str,
    content: str,
    vector: list[float],
    meta: dict | None = None,
) -> Embedding:
    """Store one embedding. Returns the persisted row."""
    row = Embedding(
        kind=kind, content=content, embedding=_validate(vector), meta=meta or {}
    )
    session.add(row)
    await session.flush()
    return row


async def add_many(
    session: AsyncSession,
    kind: str,
    items: list[tuple[str, list[float]]],
    meta: dict | None = None,
) -> int:
    """Bulk-load, used when seeding the jailbreak bank. Returns the row count."""
    rows = [
        Embedding(kind=kind, content=content, embedding=_validate(v), meta=meta or {})
        for content, v in items
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def search(
    session: AsyncSession,
    kind: str,
    vector: list[float],
    limit: int = 5,
    min_similarity: float = 0.0,
) -> list[Neighbour]:
    """
    Nearest neighbours within one `kind`, most similar first.

    `min_similarity` is applied after the index lookup rather than as a WHERE
    clause on the distance, so the ivfflat index is still used for the ordering.
    """
    _validate(vector)
    distance = Embedding.embedding.cosine_distance(vector).label("distance")

    rows = (
        await session.execute(
            select(Embedding.content, Embedding.meta, distance)
            .where(Embedding.kind == kind)
            .order_by(distance)
            .limit(limit)
        )
    ).all()

    neighbours = [
        Neighbour(content=content, similarity=1.0 - float(dist), meta=meta or {})
        for content, meta, dist in rows
    ]
    return [n for n in neighbours if n.similarity >= min_similarity]


async def count(session: AsyncSession, kind: str) -> int:
    return (
        await session.scalar(
            select(func.count(Embedding.id)).where(Embedding.kind == kind)
        )
        or 0
    )


async def clear(session: AsyncSession, kind: str) -> int:
    """Drop every vector of one kind. Used when re-seeding the bank."""
    result = await session.execute(delete(Embedding).where(Embedding.kind == kind))
    return result.rowcount or 0
