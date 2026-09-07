"""
DataLeakageDetector — PRD Detector #3
======================================
Scans user prompts AND LLM responses for sensitive data leakage:
  Layer 1: Regex + Checksum (Secrets, PII, Indian IDs)
  Layer 2: Presidio NER (Person Names, Addresses, Organisations)
  Layer 3: System Prompt Leak (N-gram overlap + Fuzzy sentence matching)

Implements the BaseDetector Protocol.

Design decisions:
  - All three layers run in sequence on every call. Layer 1 and Layer 3 are
    pure stdlib (instant). Layer 2 (Presidio) adds ~50ms per call but catches
    unstructured PII that regex cannot.
  - Deduplication: Findings from Layer 1 and Layer 2 can overlap (e.g., both
    flag the same email). We deduplicate by character span — if Layer 1 already
    flagged a span, Layer 2's duplicate is dropped (Layer 1 typically has higher
    confidence for structured patterns).
  - sub_score formula: max(individual confidences) clamped to [0, 1].
    We use max rather than mean because a single high-confidence secret leak
    is as critical as five of them.
  - System prompt leak scoring: if the prompt leak checker fires, the sub_score
    is boosted to at least 0.9 regardless of other findings, because system
    prompt exfiltration is the highest-severity data leakage event.
  - Graceful degradation: if Presidio/SpaCy are not installed, Layer 2 returns
    empty results and the detector still works with Layer 1 + Layer 3.
"""

from __future__ import annotations

from detectors.base import DetectorResult
from .patterns import PatternMatch, scan_patterns
from .presidio_scanner import NERMatch, scan_ner
from .prompt_leak import PromptLeakChecker, PromptLeakResult


class DataLeakageDetector:
    """
    Detector #3: Sensitive Data Leakage.

    Args:
        system_prompt:  The secret system prompt to monitor for leaks.
                        If None, Layer 3 (prompt leak check) is disabled.
        ner_min_score:  Minimum Presidio NER confidence (default 0.6).
    """

    detector_name: str = "data_leakage"

    def __init__(
        self,
        system_prompt: str | None = None,
        ner_min_score: float = 0.6,
    ) -> None:
        self._ner_min_score = ner_min_score
        self._leak_checker = (
            PromptLeakChecker(system_prompt) if system_prompt else None
        )

    def detect(
        self,
        prompt: str,
        response: str | None = None,
        context: list[str] | None = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Scan prompt and/or response for sensitive data leakage.

        Args:
            prompt:   User input text (always scanned).
            response: LLM output text (scanned if provided).
            context:  Unused — reserved for RAG context in other detectors.

        Returns:
            Standard DetectorResult with categorized evidence.
        """
        # Combine texts to scan
        texts_to_scan = [("prompt", prompt)]
        if response:
            texts_to_scan.append(("response", response))

        all_pattern_matches: list[dict] = []
        all_ner_matches: list[dict] = []
        prompt_leak_evidence: dict | None = None

        for source_label, text in texts_to_scan:
            # ── Layer 1: Regex + Checksum ────────────────────────────────
            pattern_hits = scan_patterns(text)
            for hit in pattern_hits:
                all_pattern_matches.append({
                    "source": source_label,
                    "category": hit.category,
                    "redacted_preview": hit.redacted_preview,
                    "confidence": hit.confidence,
                    "detection_method": hit.detection_method,
                    "start": hit.start,
                    "end": hit.end,
                })

            # ── Layer 2: Presidio NER ────────────────────────────────────
            ner_hits = scan_ner(text, min_score=self._ner_min_score)

            # Deduplicate: skip NER matches that overlap with pattern matches
            pattern_spans = {(h.start, h.end) for h in pattern_hits}
            for hit in ner_hits:
                overlaps = any(
                    hit.start < pe and hit.end > ps
                    for ps, pe in pattern_spans
                )
                if not overlaps:
                    all_ner_matches.append({
                        "source": source_label,
                        "category": hit.category,
                        "redacted_preview": hit.redacted_preview,
                        "confidence": hit.confidence,
                        "detection_method": hit.detection_method,
                        "start": hit.start,
                        "end": hit.end,
                    })

        # ── Layer 3: System Prompt Leak (response only) ──────────────────
        if response and self._leak_checker:
            leak_result = self._leak_checker.check(response)
            prompt_leak_evidence = {
                "is_leaked": leak_result.is_leaked,
                "ngram_overlap_ratio": leak_result.ngram_overlap_ratio,
                "fuzzy_max_similarity": leak_result.fuzzy_max_similarity,
                "matched_ngrams": leak_result.matched_ngrams,
                "matched_sentence": leak_result.matched_sentence,
                "detection_method": leak_result.detection_method,
            }

        # ── Aggregate results ────────────────────────────────────────────
        all_findings = all_pattern_matches + all_ner_matches
        total_findings = len(all_findings)
        has_leak = bool(prompt_leak_evidence and prompt_leak_evidence["is_leaked"])

        is_flagged = total_findings > 0 or has_leak

        # Sub-score: max confidence across all findings
        if all_findings:
            sub_score = max(f["confidence"] for f in all_findings)
        else:
            sub_score = 0.0

        # Boost sub_score if system prompt was leaked
        if has_leak:
            sub_score = max(sub_score, 0.9)

        # Overall confidence = sub_score (max-based)
        confidence = sub_score if is_flagged else 1.0 - sub_score

        return DetectorResult(
            detector_name=self.detector_name,
            is_flagged=is_flagged,
            confidence=round(confidence, 4),
            sub_score=round(sub_score, 4),
            evidence={
                "total_findings": total_findings,
                "pattern_matches": all_pattern_matches,
                "ner_matches": all_ner_matches,
                "prompt_leak": prompt_leak_evidence,
                "has_system_prompt_leak": has_leak,
            },
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton: DataLeakageDetector | None = None


def get_data_leakage_detector(
    system_prompt: str | None = None,
    **kwargs,
) -> DataLeakageDetector:
    global _singleton
    if _singleton is None:
        _singleton = DataLeakageDetector(system_prompt=system_prompt, **kwargs)
    return _singleton
