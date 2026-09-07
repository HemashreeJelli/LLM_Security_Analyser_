"""
Layer 1: Pattern Library (Rule-based Jailbreak Detection)
=========================================================
Detects known jailbreak framing techniques using regex patterns.

Techniques covered:
- DAN (Do Anything Now) & Persona adoption
- The "Grandma Exploit" (emotional manipulation)
- Hypothetical / Fictional framing
- Base64 / Leetspeak obfuscation hints

Design:
Compiled regexes for speed. Evaluates the prompt and returns all matching
techniques along with a confidence score.
"""

import re
from typing import NamedTuple

class PatternMatch(NamedTuple):
    technique: str
    confidence: float
    matched_text: str

_JAILBREAK_PATTERNS = [
    (
        "persona_adoption_dan",
        re.compile(r"(?i)\b(you are now (dan|aim|based|evil|unrestricted)|act as an unfiltered|disregard (all )?(prior )?(instructions|limitations))\b"),
        0.95
    ),
    (
        "emotional_manipulation_grandma",
        re.compile(r"(?i)\b(my (late )?(grandmother|grandma) used to|i will die if you don't|this is for a life-saving)\b"),
        0.90
    ),
    (
        "hypothetical_framing",
        re.compile(r"(?i)\b(in a hypothetical (world|scenario)|for educational purposes only|write a fictional story where|imagine a world where (rules|laws) don't apply)\b"),
        0.80
    ),
    (
        "developer_mode_override",
        re.compile(r"(?i)\b(developer mode (enabled|active)|you are in developer mode|jailbreak mode)\b"),
        0.95
    ),
    (
        "obfuscation_encoding_hint",
        re.compile(r"(?i)\b(decode the following base64|reverse this string and execute|rot13|caesar cipher|hex decoding)\b"),
        0.85
    )
]

def scan_patterns(prompt: str) -> list[PatternMatch]:
    """Scan the prompt against known jailbreak pattern templates."""
    matches = []
    for technique, pattern, confidence in _JAILBREAK_PATTERNS:
        match = pattern.search(prompt)
        if match:
            matches.append(PatternMatch(
                technique=technique,
                confidence=confidence,
                matched_text=match.group(0)
            ))
    return matches
