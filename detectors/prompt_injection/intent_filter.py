"""
Layer 2.5 — Intent Filter (Regex / Technical intent)
"""

from __future__ import annotations

import re
from typing import TypedDict


class Routing:
    CONFIRMED_INJECTION = "CONFIRMED_INJECTION"
    ROUTE_TO_JUDGE      = "ROUTE_TO_JUDGE"
    BENIGN              = "BENIGN"


class IntentResult(TypedDict):
    decision: str
    reason: str
    injection_pattern_matched: str | None
    technical_pattern_matched: str | None


_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in [
        ("ignore_instructions",
         r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)"),
        ("disregard_instructions",
         r"disregard\s+(your|all|the|any)?\s*(training|instructions?|system\s*prompt|guidelines?|rules?)"),
        ("override_instructions",
         r"override\s+(your|all|the|my)?\s*(instructions?|system\s*prompt|guidelines?|rules?)"),
        ("forget_instructions",
         r"forget\s+(all\s+)?(your|previous|prior)?\s*(instructions?|training|guidelines?|system\s*prompt)"),
        ("you_are_now",
         r"you\s+are\s+now\s+(DAN|GPT|Claude|Gemini|an?\s+unrestricted|an?\s+unfiltered|an?\s+uncensored)"),
        ("act_as_unrestricted",
         r"act\s+as\s+(if\s+you\s+(are|were)|an?)\s+(unrestricted|unfiltered|uncensored|jailbroken|evil|harmful)"),
        ("reveal_system_prompt",
         r"(reveal|print|show|output|repeat|expose|leak|display)\s+(your|the|verbatim)?\s*system\s*prompt"),
        ("fake_system_tag",
         r"(<|&lt;)\s*(system|SYSTEM|SYS|sys)\s*(>|&gt;|:)"),
        ("dan_mode",
         r"\bDAN\b.{0,40}(mode|enabled?|active|on|jailbreak)"),
    ]
]

_TECHNICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in [
        ("how_to_question",
         r"^how\s+(do\s+I|do\s+you|does|can\s+I|can\s+you|to|would\s+I|would\s+you)\s+\w"),
        ("what_is_question",
         r"^(what\s+(is|are|does|do)|explain|describe|define)\s+\w"),
        ("programming_language",
         r"\b(python|javascript|typescript|java|golang|rust|c\+\+|sql|fastapi|postgres|docker)\b"),
        ("sysadmin_terms",
         r"\b(configure|configuration|install|deploy|connection\s+pool|firewall|dns|ssl|nginx)\b"),
        ("data_science_terms",
         r"\b(machine\s+learning|deep\s+learning|neural\s+network|transformer|embedding|vector)\b"),
    ]
]


def _first_match(text: str, patterns: list[tuple[str, re.Pattern]]) -> str | None:
    for name, rx in patterns:
        if rx.search(text):
            return name
    return None


class IntentFilter:
    def __init__(self) -> None:
        self._injection = list(_INJECTION_PATTERNS)
        self._technical = list(_TECHNICAL_PATTERNS)

    def decide(self, text: str, layer2_label: str, injection_score: float) -> IntentResult:
        inj_hit = _first_match(text, self._injection)
        tech_hit = _first_match(text, self._technical)

        if inj_hit:
            return IntentResult(
                decision=Routing.CONFIRMED_INJECTION,
                reason=f"Injection pattern '{inj_hit}' matched. Score: {injection_score:.3f}",
                injection_pattern_matched=inj_hit,
                technical_pattern_matched=tech_hit,
            )

        if layer2_label == "INJECTION" and tech_hit:
            return IntentResult(
                decision=Routing.ROUTE_TO_JUDGE,
                reason=f"Layer 2 flagged ({injection_score:.3f}) but tech pattern '{tech_hit}' matched.",
                injection_pattern_matched=None,
                technical_pattern_matched=tech_hit,
            )

        if layer2_label == "INJECTION":
            return IntentResult(
                decision=Routing.CONFIRMED_INJECTION,
                reason=f"Layer 2 flagged ({injection_score:.3f}) with no technical pattern.",
                injection_pattern_matched=None,
                technical_pattern_matched=None,
            )

        return IntentResult(
            decision=Routing.BENIGN,
            reason=f"Layer 2 score {injection_score:.3f} (BENIGN), no injection pattern matched.",
            injection_pattern_matched=None,
            technical_pattern_matched=tech_hit,
        )


_singleton: IntentFilter | None = None


def get_filter() -> IntentFilter:
    global _singleton
    if _singleton is None:
        _singleton = IntentFilter()
    return _singleton
