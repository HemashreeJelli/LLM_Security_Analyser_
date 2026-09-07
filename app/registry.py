"""
Detector registry with graceful degradation.

The detector modules have heavyweight requirements: two fine-tuned transformer
checkpoints (~1 GB combined) that are excluded from git, a spaCy model for
Presidio, and an LLM-judge API key. Any of those can be absent on a given
machine.

The registry's job is to make that a reported condition instead of a crash. It
attempts to construct every detector at startup, keeps the ones that came up,
and records why each of the others did not. The service then runs with whatever
loaded, and every response carries a `degraded` list naming what was missing —
so a low risk score is never mistaken for a clean verdict when half the suite
failed to load.

This is also what lets the platform be developed and demonstrated
independently of the detector work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import get_settings

log = logging.getLogger(__name__)

# Every detector named in the PRD. Ones that are not yet implemented appear in
# `unavailable` with a "not implemented" reason, which keeps the gap visible in
# /health rather than letting it disappear silently.
ALL_DETECTORS = (
    "prompt_injection",
    "jailbreak",
    "data_leakage",
    "hallucination",
    "unsafe_output",
)


class SyncDetector(Protocol):
    detector_name: str

    def detect(
        self,
        prompt: str,
        response: str | None = ...,
        context: list[str] | None = ...,
        **kwargs: Any,
    ) -> dict: ...


@dataclass
class Registry:
    loaded: dict[str, SyncDetector] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.loaded)

    @property
    def is_degraded(self) -> bool:
        return bool(self.unavailable)


_registry: Registry | None = None


def _build_judge_backend():
    """
    Shared LLM-judge backend for the detectors that escalate grey-zone cases.

    Returns None when no key is configured; the detectors accept a None backend
    and simply skip their judge layer.
    """
    import os

    if not os.getenv("GROQ_API_KEY"):
        log.info("GROQ_API_KEY not set - LLM judge layer disabled")
        return None

    from llm_judge.backends.groq_backend import GroqBackend

    return GroqBackend(model="groq/compound")


def build_registry() -> Registry:
    """
    Construct every available detector. Called once at application startup so
    that model loading happens before the first request rather than inside it.
    """
    settings = get_settings()
    reg = Registry()

    try:
        backend = _build_judge_backend()
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM judge backend unavailable: %s", exc)
        backend = None

    def attempt(name: str, factory) -> None:
        try:
            reg.loaded[name] = factory()
            log.info("detector loaded: %s", name)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            reg.unavailable[name] = reason
            log.warning("detector unavailable: %s (%s)", name, reason)

    def _prompt_injection():
        from detectors.prompt_injection.detector import PromptInjectionDetector

        return PromptInjectionDetector(judge_backend=backend)

    def _jailbreak():
        from detectors.jailbreak import get_jailbreak_detector

        return get_jailbreak_detector(backend=backend)

    def _data_leakage():
        from detectors.data_leakage import get_data_leakage_detector

        return get_data_leakage_detector(system_prompt=settings.system_prompt)

    attempt("prompt_injection", _prompt_injection)
    attempt("jailbreak", _jailbreak)
    attempt("data_leakage", _data_leakage)

    for name in ALL_DETECTORS:
        if name not in reg.loaded and name not in reg.unavailable:
            reg.unavailable[name] = "not implemented yet"

    log.info(
        "registry ready: %d loaded, %d unavailable",
        len(reg.loaded),
        len(reg.unavailable),
    )
    return reg


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = build_registry()
    return _registry


def reset_registry() -> None:
    """Drop the cached registry. Used by tests."""
    global _registry
    _registry = None
