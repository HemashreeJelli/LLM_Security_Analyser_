"""
Concurrent detector execution (PRD Section 1).

The detectors are synchronous and CPU/IO-heavy, so each runs in a worker thread
and the whole set is awaited together. That keeps end-to-end latency close to
the slowest single detector rather than the sum of all of them, without pulling
in an external task queue.

Two failure modes are handled here so they never reach the caller as a 500:

  timeout — a detector that overruns detector_timeout_seconds is abandoned and
            recorded as errored. One slow LLM-judge call cannot hold the
            request open indefinitely.
  crash   — gather runs with return_exceptions=True, and any exception becomes
            an errored report. The scoring engine treats errored detectors as
            contributing nothing, so the composite degrades instead of failing.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.config import get_settings
from app.registry import Registry
from app.schemas import AnalyzeRequest, DetectorReport

log = logging.getLogger(__name__)


def _errored(name: str, reason: str, latency_ms: int = 0) -> DetectorReport:
    """A neutral report standing in for a detector that could not produce one."""
    return DetectorReport(
        detector_name=name,
        is_flagged=False,
        confidence=0.0,
        sub_score=0.0,
        evidence={},
        latency_ms=latency_ms,
        error=reason,
    )


def _coerce(name: str, raw: dict, latency_ms: int) -> DetectorReport:
    """
    Validate a detector's dict against the shared contract.

    A detector returning a malformed payload is a bug on the detector side, but
    it should surface as one errored detector rather than a failed request.
    """
    try:
        return DetectorReport(
            detector_name=raw.get("detector_name", name),
            is_flagged=bool(raw["is_flagged"]),
            confidence=float(raw["confidence"]),
            sub_score=float(raw["sub_score"]),
            evidence=raw.get("evidence") or {},
            latency_ms=latency_ms,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _errored(name, f"malformed DetectorResult: {exc}", latency_ms)


async def _run_one(
    name: str, detector, request: AnalyzeRequest, timeout: float
) -> DetectorReport:
    started = time.perf_counter()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                detector.detect,
                prompt=request.prompt,
                response=request.response,
                context=request.context_chunks or None,
                conversation_history=[t.model_dump() for t in request.history] or None,
                system_prompt=request.system_prompt,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = int((time.perf_counter() - started) * 1000)
        log.warning("detector %s timed out after %.1fs", name, timeout)
        return _errored(name, f"timed out after {timeout}s", elapsed)
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - started) * 1000)
        log.exception("detector %s raised", name)
        return _errored(name, f"{type(exc).__name__}: {exc}", elapsed)

    elapsed = int((time.perf_counter() - started) * 1000)
    return _coerce(name, raw, elapsed)


async def run_detectors(
    request: AnalyzeRequest, registry: Registry
) -> list[DetectorReport]:
    """Run every loaded detector concurrently and return their reports."""
    timeout = get_settings().detector_timeout_seconds

    tasks = [
        _run_one(name, detector, request, timeout)
        for name, detector in registry.loaded.items()
    ]
    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)

    reports: list[DetectorReport] = []
    for name, result in zip(registry.loaded, results):
        if isinstance(result, BaseException):
            # _run_one already traps detector errors, so reaching here means the
            # orchestration itself failed. Recorded rather than raised.
            reports.append(_errored(name, f"orchestration error: {result}"))
        else:
            reports.append(result)
    return reports
