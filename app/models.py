"""
Persistence schema (PRD Section 12).

Four tables:
  analyses          — one row per /analyze call: the inputs, the composite
                      verdict, and the recommendation. This is the audit log
                      and the source for dashboard trend queries.
  detector_results  — one row per detector per analysis, holding the full
                      evidence blob. Split out from `analyses` so that
                      per-detector metrics can be computed with a GROUP BY
                      instead of unpacking JSON in the application.
  api_keys          — issued keys, stored as SHA-256 digests, never plaintext.
  embeddings        — pgvector store. Backs the jailbreak similarity bank and
                      the groundedness lookups the hallucination detector will
                      need, so both share one index and one extension.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # ── Inputs (retained for audit and for replaying the eval harness) ────
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    history: Mapped[list] = mapped_column(JSONB, default=list)
    context_chunks: Mapped[list] = mapped_column(JSONB, default=list)

    # ── Verdict ──────────────────────────────────────────────────────────
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    recommendation: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    degraded: Mapped[list] = mapped_column(JSONB, default=list)

    # ── Provenance ───────────────────────────────────────────────────────
    api_key_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    detector_results: Mapped[list["DetectorResultRow"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )


class DetectorResultRow(Base):
    __tablename__ = "detector_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )

    detector_name: Mapped[str] = mapped_column(String(64), index=True)
    is_flagged: Mapped[bool] = mapped_column(Boolean, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    sub_score: Mapped[float] = mapped_column(Float)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis: Mapped[Analysis] = relationship(back_populates="detector_results")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    label: Mapped[str] = mapped_column(String(64), unique=True)
    # SHA-256 of the key. The plaintext is shown once at issue time and never
    # stored, so a database dump does not hand over working credentials.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


class Embedding(Base):
    """
    Shared vector store.

    `kind` partitions the table by purpose ("jailbreak_bank" for the curated
    attack-prompt bank, "context_chunk" for RAG groundedness) so that one
    table and one ivfflat index serve both detectors.
    """

    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kind: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dim)
    )
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


# Cosine-distance index. lists=100 suits the few-thousand-row bank described
# in the PRD; it should be retuned if the bank grows past ~100k rows.
Index(
    "ix_embeddings_vector_cosine",
    Embedding.embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 100},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
