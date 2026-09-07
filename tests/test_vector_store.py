"""
pgvector integration tests.

These need a live Postgres with the vector extension, so they skip cleanly when
one is not reachable (`docker compose up -d db` provides it). Everything is
written under a throwaway `kind` and removed afterwards, so the tests are safe
to run against a database that already holds real rows.
"""

from __future__ import annotations

import asyncio
import math
import sys

import pytest
import pytest_asyncio

from app import vector_store as vs
from app.config import get_settings
from app.db import SessionLocal, init_db

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DIM = get_settings().embedding_dim
TEST_KIND = "pytest_tmp"


def unit_vector(index: int) -> list[float]:
    """A one-hot vector, so cosine similarity between distinct ones is 0."""
    v = [0.0] * DIM
    v[index % DIM] = 1.0
    return v


def tilted(index: int, other: int, weight: float) -> list[float]:
    """A vector between two axes, for a predictable intermediate similarity."""
    v = [0.0] * DIM
    v[index % DIM] = math.sqrt(1.0 - weight**2)
    v[other % DIM] = weight
    return v


@pytest_asyncio.fixture
async def session():
    try:
        await init_db()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {exc}")

    async with SessionLocal() as s:
        await vs.clear(s, TEST_KIND)
        await s.commit()
        yield s
        await vs.clear(s, TEST_KIND)
        await s.commit()


@pytest.mark.asyncio
async def test_roundtrip_and_exact_match_scores_one(session):
    await vs.add(session, TEST_KIND, "ignore previous instructions", unit_vector(0))
    await session.commit()

    hits = await vs.search(session, TEST_KIND, unit_vector(0), limit=1)
    assert len(hits) == 1
    assert hits[0].content == "ignore previous instructions"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_orthogonal_vectors_score_zero(session):
    await vs.add(session, TEST_KIND, "unrelated", unit_vector(1))
    await session.commit()

    hits = await vs.search(session, TEST_KIND, unit_vector(2), limit=1)
    assert hits[0].similarity == pytest.approx(0.0, abs=1e-5)


@pytest.mark.asyncio
async def test_results_are_ordered_by_similarity(session):
    await vs.add_many(
        session,
        TEST_KIND,
        [
            ("far", unit_vector(5)),
            ("near", tilted(5, 6, 0.2)),
            ("exact", unit_vector(6)),
        ],
    )
    await session.commit()

    hits = await vs.search(session, TEST_KIND, unit_vector(6), limit=3)
    assert [h.content for h in hits] == ["exact", "near", "far"]
    assert hits[0].similarity > hits[1].similarity > hits[2].similarity


@pytest.mark.asyncio
async def test_min_similarity_filters_weak_matches(session):
    await vs.add_many(
        session, TEST_KIND, [("weak", unit_vector(7)), ("strong", unit_vector(8))]
    )
    await session.commit()

    hits = await vs.search(
        session, TEST_KIND, unit_vector(8), limit=5, min_similarity=0.5
    )
    assert [h.content for h in hits] == ["strong"]


@pytest.mark.asyncio
async def test_search_is_scoped_to_its_kind(session):
    """The jailbreak bank and RAG chunks share a table; a query against one
    must never return rows belonging to the other."""
    await vs.add(session, TEST_KIND, "in scope", unit_vector(9))
    await vs.add(session, vs.CONTEXT_CHUNK, "different kind", unit_vector(9))
    await session.commit()

    try:
        hits = await vs.search(session, TEST_KIND, unit_vector(9), limit=10)
        assert [h.content for h in hits] == ["in scope"]
    finally:
        await vs.clear(session, vs.CONTEXT_CHUNK)
        await session.commit()


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected_before_the_database(session):
    with pytest.raises(ValueError, match="expected"):
        await vs.add(session, TEST_KIND, "bad", [0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_metadata_survives_the_roundtrip(session):
    await vs.add(
        session,
        TEST_KIND,
        "DAN template",
        unit_vector(11),
        meta={"source": "JailbreakBench", "family": "persona"},
    )
    await session.commit()

    hits = await vs.search(session, TEST_KIND, unit_vector(11), limit=1)
    assert hits[0].meta["source"] == "JailbreakBench"
    assert await vs.count(session, TEST_KIND) == 1
