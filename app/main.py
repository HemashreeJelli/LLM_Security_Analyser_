"""
FastAPI application (PRD Section 1).

One asynchronous service exposing /analyze, backed by Postgres. Detector
loading and schema creation both happen in the lifespan handler so that model
weights and the pgvector extension are ready before the first request is
served, which keeps the first call out of the latency percentiles.

Startup is deliberately fault-tolerant: neither an unreachable database nor a
missing detector prevents the service from coming up. Both conditions are
reported by /health and, for detectors, on every analysis response. A demo or a
developer machine should be able to run the API without the full stack present.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone


from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import Integer, desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app import db as database
from app.auth import require_api_key
from app.config import get_settings
from app.db import get_session, init_db
from app.mitigation import recommend
from app.models import Analysis, DetectorResultRow
from app.orchestrator import run_detectors
from app.registry import get_registry
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.scoring import score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("llmsec")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s (%s)", settings.app_name, settings.environment)

    # A missing database degrades the service; it must never prevent it from
    # starting. init_db is internally bounded by a timeout, so an unreachable
    # host raises here in seconds rather than stalling startup forever.
    try:
        await init_db()
        app.state.db_ready = True
    except Exception as exc:  # noqa: BLE001
        log.warning("database unavailable at startup: %s", exc)
        log.warning(
            "running without persistence - /analyze works, results are not stored"
        )
        app.state.db_ready = False

    # Warms every detector, including model loading. This must go through
    # get_registry() rather than build_registry(): the routes read the cached
    # registry, so building a detached one here would leave the first request
    # to construct its own - paying the full model-load cost inside the request
    # that startup was supposed to have absorbed.
    app.state.registry = get_registry()

    yield

    await database.engine.dispose()
    log.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
    description=(
        "Security assessment for LLM interactions. Submit a prompt/response "
        "pair to /analyze and receive a composite risk score, per-detector "
        "evidence, and a mitigation recommendation."
    ),
    lifespan=lifespan,
)

# The dashboard is served from a separate origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def database_unavailable(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    Turn a dead database into an honest 503 on the read endpoints.

    /analyze handles its own persistence failure and still returns a verdict,
    so it never reaches here. The history and stats endpoints have nothing to
    serve without Postgres, and saying so beats a 500 the dashboard renders as
    "cannot reach the API".
    """
    log.error("database error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Database unavailable. The analyzer still runs, but stored "
                "history and statistics need Postgres - try `docker compose up -d db`."
            )
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    registry = get_registry()
    db_up = await database.ping()
    return HealthResponse(
        status="ok" if db_up and not registry.is_degraded else "degraded",
        environment=settings.environment,
        database="up" if db_up else "down",
        auth_enabled=settings.auth_enabled,
        detectors_loaded=registry.names,
        detectors_unavailable=registry.unavailable,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(
    request: AnalyzeRequest,
    session: AsyncSession = Depends(get_session),
    caller: str = Depends(require_api_key),
) -> AnalyzeResponse:
    """
    Run every available detector over the interaction, aggregate the results
    into a composite risk score, and return a mitigation recommendation.
    """
    started = time.perf_counter()
    registry = get_registry()

    reports = await run_detectors(request, registry)

    # A detector absent from this deployment and one that failed mid-run are the
    # same thing from the caller's point of view: a gap in coverage.
    degraded = sorted(
        set(registry.unavailable) | {r.detector_name for r in reports if r.error}
    )

    result = score(reports)
    recommendation = recommend(reports, result.severity, degraded)
    latency_ms = int((time.perf_counter() - started) * 1000)

    row = Analysis(
        prompt=request.prompt,
        response=request.response,
        history=[t.model_dump() for t in request.history],
        context_chunks=request.context_chunks,
        risk_score=result.risk_score,
        severity=result.severity,
        recommendation=recommendation,
        latency_ms=latency_ms,
        degraded=degraded,
        api_key_label=caller,
        detector_results=[
            DetectorResultRow(
                detector_name=r.detector_name,
                is_flagged=r.is_flagged,
                confidence=r.confidence,
                sub_score=r.sub_score,
                evidence=r.evidence,
                latency_ms=r.latency_ms,
                error=r.error,
            )
            for r in reports
        ],
    )

    created_at = datetime.now(timezone.utc)
    request_id = str(uuid.uuid4())

    try:
        session.add(row)
        await session.flush()
        request_id = str(row.id)
        created_at = row.created_at or created_at
    except Exception as exc:  # noqa: BLE001
        # Losing the audit row must not lose the analysis the caller asked for.
        await session.rollback()
        log.error("failed to persist analysis: %s", exc)

    return AnalyzeResponse(
        request_id=request_id,
        risk_score=result.risk_score,
        severity=result.severity,
        detectors=reports,
        recommendation=recommendation,
        latency_ms=latency_ms,
        created_at=created_at,
        degraded=degraded,
    )


@app.get("/analyses", tags=["analysis"])
async def list_analyses(
    limit: int = Query(50, ge=1, le=500),
    severity: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
    caller: str = Depends(require_api_key),
) -> dict:
    """Recent analyses, newest first. Backs the dashboard activity feed."""
    stmt = select(Analysis).order_by(desc(Analysis.created_at)).limit(limit)
    if severity:
        stmt = stmt.where(Analysis.severity == severity)

    rows = (await session.scalars(stmt)).all()
    return {
        "count": len(rows),
        "items": [
            {
                "request_id": str(r.id),
                "created_at": r.created_at,
                "prompt": r.prompt[:200],
                "risk_score": r.risk_score,
                "severity": r.severity,
                "latency_ms": r.latency_ms,
                "flagged_detectors": [
                    d.detector_name for d in r.detector_results if d.is_flagged
                ],
            }
            for r in rows
        ],
    }


@app.get("/analyses/{analysis_id}", tags=["analysis"])
async def get_analysis(
    analysis_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    caller: str = Depends(require_api_key),
) -> dict:
    """Full stored record including every detector's evidence."""
    row = await session.get(Analysis, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return {
        "request_id": str(row.id),
        "created_at": row.created_at,
        "prompt": row.prompt,
        "response": row.response,
        "risk_score": row.risk_score,
        "severity": row.severity,
        "recommendation": row.recommendation,
        "latency_ms": row.latency_ms,
        "degraded": row.degraded,
        "detectors": [
            {
                "detector_name": d.detector_name,
                "is_flagged": d.is_flagged,
                "confidence": d.confidence,
                "sub_score": d.sub_score,
                "evidence": d.evidence,
                "latency_ms": d.latency_ms,
                "error": d.error,
            }
            for d in row.detector_results
        ],
    }


@app.get("/stats", tags=["analysis"])
async def stats(
    session: AsyncSession = Depends(get_session),
    caller: str = Depends(require_api_key),
) -> dict:
    """
    Aggregate counters for the dashboard header: volume, severity mix, latency,
    and per-detector flag rates.
    """
    total = await session.scalar(select(func.count(Analysis.id))) or 0

    by_severity = dict(
        (
            await session.execute(
                select(Analysis.severity, func.count(Analysis.id)).group_by(
                    Analysis.severity
                )
            )
        ).all()
    )

    detector_rows = (
        await session.execute(
            select(
                DetectorResultRow.detector_name,
                func.count(DetectorResultRow.id),
                func.sum(func.cast(DetectorResultRow.is_flagged, Integer)),
                func.avg(DetectorResultRow.latency_ms),
            ).group_by(DetectorResultRow.detector_name)
        )
    ).all()

    by_detector = [
        {
            "detector_name": name,
            "runs": runs,
            "flagged": flagged or 0,
            "flag_rate": round((flagged or 0) / runs, 4) if runs else 0.0,
            "avg_latency_ms": round(float(avg_latency or 0.0), 1),
        }
        for name, runs, flagged, avg_latency in detector_rows
    ]

    latency = await session.scalar(select(func.avg(Analysis.latency_ms)))

    return {
        "total_analyses": total,
        "by_severity": by_severity,
        "by_detector": by_detector,
        "avg_latency_ms": round(float(latency or 0.0), 1),
    }
