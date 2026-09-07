"""
Layer 2 — Presidio NER Scanner
===============================
Wraps Microsoft Presidio's AnalyzerEngine to detect unstructured PII
that regex cannot catch: person names, physical addresses, organisations.

Also registers custom Indian recognizers (Aadhaar, PAN) into Presidio's
registry so they participate in Presidio's context-enhancement and
conflict-resolution pipeline alongside the built-in recognizers.

Design decisions:
  - Lazy init: AnalyzerEngine is heavy (~2s to load SpaCy model).
    We use a module-level singleton so it loads once on first call.
  - Threshold filtering: Presidio can return low-confidence noise
    (e.g., tagging "Python" as a PERSON). We apply a configurable
    minimum score threshold (default 0.6) before returning results.
  - Graceful degradation: if presidio or spacy are not installed,
    this layer returns an empty list instead of crashing the pipeline.
  - Custom Indian recognizers are added via Presidio's PatternRecognizer
    API — no model training needed, just regex + context keywords.
"""

from __future__ import annotations

from typing import NamedTuple


class NERMatch(NamedTuple):
    category: str          # Presidio entity type: PERSON, LOCATION, etc.
    value: str             # matched text
    redacted_preview: str  # "[PERSON]"
    start: int
    end: int
    confidence: float
    detection_method: str  # "presidio_ner"


def _build_analyzer():
    """
    Build and configure the Presidio AnalyzerEngine with custom Indian
    recognizers. Returns None if presidio/spacy not available.
    """
    try:
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    except ImportError:
        return None

    analyzer = AnalyzerEngine()

    # ── Custom Indian Recognizers ────────────────────────────────────────

    # Aadhaar (12-digit, starts with 2-9)
    aadhaar_recognizer = PatternRecognizer(
        supported_entity="IN_AADHAAR",
        name="Indian Aadhaar Recognizer",
        patterns=[Pattern(
            name="aadhaar_pattern",
            regex=r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b",
            score=0.5,  # base score — boosted by context
        )],
        context=["aadhaar", "aadhar", "uid", "uidai", "unique identification"],
        supported_language="en",
    )

    # PAN Card (5 letters + 4 digits + 1 letter)
    pan_recognizer = PatternRecognizer(
        supported_entity="IN_PAN",
        name="Indian PAN Card Recognizer",
        patterns=[Pattern(
            name="pan_pattern",
            regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            score=0.5,
        )],
        context=["pan", "pan card", "permanent account number", "income tax"],
        supported_language="en",
    )

    # Indian Phone (+91)
    in_phone_recognizer = PatternRecognizer(
        supported_entity="IN_PHONE",
        name="Indian Phone Number Recognizer",
        patterns=[Pattern(
            name="in_phone_pattern",
            regex=r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b",
            score=0.4,
        )],
        context=["phone", "mobile", "contact", "call", "whatsapp"],
        supported_language="en",
    )

    # IFSC Code
    ifsc_recognizer = PatternRecognizer(
        supported_entity="IN_IFSC",
        name="Indian IFSC Code Recognizer",
        patterns=[Pattern(
            name="ifsc_pattern",
            regex=r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            score=0.4,
        )],
        context=["ifsc", "bank", "neft", "rtgs", "imps", "branch"],
        supported_language="en",
    )

    # Register all custom recognizers
    registry = analyzer.registry
    registry.add_recognizer(aadhaar_recognizer)
    registry.add_recognizer(pan_recognizer)
    registry.add_recognizer(in_phone_recognizer)
    registry.add_recognizer(ifsc_recognizer)

    return analyzer


# Module-level singleton
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = _build_analyzer()
    return _analyzer


def scan_ner(
    text: str,
    min_score: float = 0.6,
    entities: list[str] | None = None,
) -> list[NERMatch]:
    """
    Scan text using Presidio NER engine for unstructured PII.

    Args:
        text:      Input text to scan.
        min_score: Minimum confidence threshold (default 0.6).
        entities:  Optional whitelist of entity types to detect.
                   If None, detect all supported entities.

    Returns:
        List of NERMatch results above the threshold.
    """
    analyzer = _get_analyzer()
    if analyzer is None:
        return []  # graceful degradation

    results = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
    )

    matches: list[NERMatch] = []
    for r in results:
        if r.score < min_score:
            continue
        value = text[r.start:r.end]
        matches.append(NERMatch(
            category=r.entity_type,
            value=value,
            redacted_preview=f"[{r.entity_type}]",
            start=r.start,
            end=r.end,
            confidence=round(r.score, 4),
            detection_method="presidio_ner",
        ))

    matches.sort(key=lambda m: m.start)
    return matches
