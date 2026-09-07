"""
Jailbreak Detector Package.

Implements PRD Section 6.2:
- Layer 1: Pattern library (DAN, Grandma, Hypothetical, Encodings).
- Layer 2: Embedding-similarity check against known jailbreak banks.
- Layer 3: Multi-turn escalation tracking.
"""

from .detector import JailbreakDetector, get_jailbreak_detector

__all__ = ["JailbreakDetector", "get_jailbreak_detector"]
