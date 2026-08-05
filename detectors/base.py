"""
BaseDetector — Shared contract for all 5 security detectors.

PRD Section 14:
  "The Detector interface contract — the exact input/output shape
   every detector module must implement ({label, confidence, evidence, sub_score})."
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class DetectorResult(TypedDict):
    detector_name: str     # e.g. "prompt_injection", "jailbreak", "data_leakage"
    is_flagged: bool        # True if attack/risk detected above threshold
    confidence: float       # Confidence of the detector [0.0 - 1.0]
    sub_score: float        # Normalized risk sub-score s_i ∈ [0.0, 1.0]
    evidence: dict          # Detailed evidence dictionary for UI drill-down


@runtime_checkable
class BaseDetector(Protocol):
    """Protocol that all detector modules must satisfy."""

    detector_name: str

    def detect(
        self,
        prompt: str,
        response: str | None = None,
        context: list[str] | None = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Analyze the input and return a DetectorResult.

        Args:
            prompt:   User prompt / input string.
            response: Optional model response (for output detectors).
            context:  Optional RAG context chunks.

        Returns:
            DetectorResult TypedDict.
        """
        ...
