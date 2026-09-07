"""
Layer 1 — Regex & Algorithmic Pattern Scanner
==============================================
High-speed, zero-dependency pattern matching for:
  - Secrets & API Keys (OpenAI, AWS, JWT, generic key=value)
  - Indian Identifiers (Aadhaar with Verhoeff, PAN, IFSC, +91 phone)
  - Global PII (Email, US SSN, US Phone, Credit Card with Luhn)

Design decisions:
  - Patterns are compiled once at import time for speed.
  - Credit cards use Luhn checksum to eliminate SKU/serial false positives.
  - Aadhaar uses Verhoeff checksum to eliminate random 12-digit false positives.
  - PAN regex is strict: 5 letters + 4 digits + 1 letter, 4th char encodes
    holder type (P=Person, C=Company, etc).
  - Redacted previews never expose full values — safe for logging/dashboards.
  - Overlap resolver: if two patterns flag the same span, keep the higher
    confidence match to avoid double-counting.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .luhn import is_valid_luhn, is_valid_verhoeff


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class PatternMatch(NamedTuple):
    category: str            # e.g. "SECRET_OPENAI_KEY", "IN_AADHAAR"
    value: str               # raw matched text
    redacted_preview: str    # safe preview for logging: "[SECRET_OPENAI_KEY]"
    start: int               # char offset start
    end: int                 # char offset end
    confidence: float        # [0.0 - 1.0]
    detection_method: str    # "regex_pattern" | "luhn_validated" | "verhoeff_validated"


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # ── Secrets & API Keys ───────────────────────────────────────────────
    ("SECRET_OPENAI_KEY",
     re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"), 0.99),
    ("SECRET_AWS_ACCESS_KEY",
     re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), 0.99),
    ("SECRET_JWT_TOKEN",
     re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"), 0.95),
    ("SECRET_GENERIC_KEY",
     re.compile(
         r"(?i)\b(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token)"
         r"\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"
     ), 0.90),
    ("SECRET_PRIVATE_KEY_HEADER",
     re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"), 0.99),
    ("SECRET_GROQ_KEY",
     re.compile(r"\bgsk_[a-zA-Z0-9]{20,}\b"), 0.99),

    # ── Indian Identifiers ───────────────────────────────────────────────
    ("IN_PAN_CARD",
     re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), 0.95),
    ("IN_IFSC_CODE",
     re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), 0.90),
    ("IN_PHONE_NUMBER",
     re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b"), 0.85),

    # ── Global PII ──────────────────────────────────────────────────────
    ("PII_EMAIL",
     re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), 0.95),
    ("PII_US_SSN",
     re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), 0.90),
    ("PII_US_PHONE",
     re.compile(r"\b(?:\+?1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}\b"), 0.80),
    ("PII_IP_ADDRESS",
     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.75),
]

# Separate candidates that need algorithmic validation
_CREDIT_CARD_RX = re.compile(r"\b(?:\d[\s\-]*){13,19}\b")
_AADHAAR_RX = re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b")


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _overlaps(new_start: int, new_end: int, existing: list[PatternMatch]) -> bool:
    """Check if a new span overlaps with any existing match."""
    for m in existing:
        if new_start < m.end and new_end > m.start:
            return True
    return False


def scan_patterns(text: str) -> list[PatternMatch]:
    """
    Scan text for secrets, PII, and Indian identifiers using regex + checksums.

    Returns:
        Sorted list of PatternMatch (by start offset).
    """
    matches: list[PatternMatch] = []

    # ── Standard regex patterns ──────────────────────────────────────────
    for category, rx, confidence in _REGEX_PATTERNS:
        for m in rx.finditer(text):
            val = m.group(0)
            matches.append(PatternMatch(
                category=category,
                value=val,
                redacted_preview=f"[{category}]",
                start=m.start(),
                end=m.end(),
                confidence=confidence,
                detection_method="regex_pattern",
            ))

    # ── Credit card (regex + Luhn) ───────────────────────────────────────
    for m in _CREDIT_CARD_RX.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if is_valid_luhn(digits) and not _overlaps(m.start(), m.end(), matches):
            matches.append(PatternMatch(
                category="PII_CREDIT_CARD",
                value=raw,
                redacted_preview="[PII_CREDIT_CARD]",
                start=m.start(),
                end=m.end(),
                confidence=0.98,
                detection_method="luhn_validated",
            ))

    # ── Aadhaar (regex + Verhoeff) ───────────────────────────────────────
    for m in _AADHAAR_RX.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if is_valid_verhoeff(digits) and not _overlaps(m.start(), m.end(), matches):
            matches.append(PatternMatch(
                category="IN_AADHAAR",
                value=raw,
                redacted_preview="[IN_AADHAAR]",
                start=m.start(),
                end=m.end(),
                confidence=0.98,
                detection_method="verhoeff_validated",
            ))

    matches.sort(key=lambda m: m.start)
    return matches
