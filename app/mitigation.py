"""
Mitigation recommender (PRD Section 8).

Deliberately a lookup table rather than a learned component. Levels 1 and 2
(detection and scoring) use ML; this stage stays rules so that the guidance an
analyst reads is always traceable to a rule someone wrote and can argue with.
That explainability is the whole point of keeping it dumb.

The table is keyed on (detector, severity band). Recommendations are assembled
from the highest-severity finding downward, so the most urgent action is the
first line an analyst sees.
"""

from __future__ import annotations

from app.schemas import DetectorReport, Severity

_BAND_ORDER: dict[Severity, int] = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}

# (detector_name, severity) -> operator-facing action.
_PLAYBOOK: dict[tuple[str, Severity], str] = {
    ("prompt_injection", "Medium"): (
        "Log the interaction and re-assert the system prompt on the next turn."
    ),
    ("prompt_injection", "High"): (
        "Block the prompt. Strip the flagged span, re-run with delimiters escaped, "
        "and require the model to restate its instructions before continuing."
    ),
    ("prompt_injection", "Critical"): (
        "Block and terminate the session. The instruction hierarchy was overridden — "
        "rotate any credentials reachable from this agent's tool scope and review "
        "the last 24h of traffic from this API key for the same pattern."
    ),
    ("jailbreak", "Medium"): (
        "Log and monitor. Track this conversation for multi-turn escalation."
    ),
    ("jailbreak", "High"): (
        "Refuse the request and reset the conversation context. Persona-adoption "
        "framing survives across turns, so clearing history matters more than "
        "blocking the single prompt."
    ),
    ("jailbreak", "Critical"): (
        "Block, terminate the session, and rate-limit the caller. Add the prompt to "
        "the jailbreak bank so similarity matching catches variants of it."
    ),
    ("data_leakage", "Medium"): (
        "Redact the flagged entities before the response reaches the user."
    ),
    ("data_leakage", "High"): (
        "Block the response and redact. Confirm the exposed values are synthetic; "
        "if not, treat as a data incident."
    ),
    ("data_leakage", "Critical"): (
        "Block the response and open an incident. System-prompt or credential "
        "material left the model — rotate the exposed secrets immediately and "
        "audit prior responses to the same caller."
    ),
    ("hallucination", "Medium"): (
        "Attach a low-confidence notice and cite the retrieved chunks used."
    ),
    ("hallucination", "High"): (
        "Withhold the response and re-query with tighter retrieval. Ungrounded "
        "claims were asserted that the context does not support."
    ),
    ("hallucination", "Critical"): (
        "Block the response. Claims directly contradict the retrieved context — "
        "check whether the retrieval index is stale or the wrong corpus is mounted."
    ),
    ("unsafe_output", "Medium"): (
        "Apply the standard content filter before delivery."
    ),
    ("unsafe_output", "High"): (
        "Block the response and return the refusal template."
    ),
    ("unsafe_output", "Critical"): (
        "Block, log for trust-and-safety review, and suspend the caller pending "
        "review. Route to a human if the category is self-harm."
    ),
}

_CLEAN = "No action required. All detectors returned within normal thresholds."

_DEGRADED_NOTE = (
    "Detectors unavailable in this deployment ({names}) did not contribute to this "
    "score. Treat the verdict as a partial assessment."
)


def _band_for(sub_score: float) -> Severity:
    if sub_score <= 0.2:
        return "Low"
    if sub_score <= 0.5:
        return "Medium"
    if sub_score <= 0.8:
        return "High"
    return "Critical"


def recommend(
    reports: list[DetectorReport],
    overall: Severity,
    degraded: list[str] | None = None,
) -> str:
    """
    Build the recommendation text for one analysis.

    Only flagged detectors contribute lines, ordered most severe first. When
    the composite is Low but individual detectors still fired, their advisory
    lines are kept — a Low composite with a Medium leakage finding is still
    worth telling the analyst about.
    """
    lines: list[str] = []

    flagged = [r for r in reports if r.is_flagged and r.error is None]
    ranked = sorted(
        flagged,
        key=lambda r: (_BAND_ORDER[_band_for(r.sub_score)], r.sub_score),
        reverse=True,
    )

    for report in ranked:
        band = _band_for(report.sub_score)
        action = _PLAYBOOK.get((report.detector_name, band))
        if action:
            lines.append(f"[{band}] {report.detector_name}: {action}")

    if not lines:
        lines.append(_CLEAN)

    if degraded:
        lines.append(_DEGRADED_NOTE.format(names=", ".join(sorted(degraded))))

    return "\n".join(lines)
