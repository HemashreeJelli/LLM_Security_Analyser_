"""
Risk-scoring engine (PRD Section 7).

Takes the five per-detector sub-scores and folds them into one 0-100 composite
plus a severity band.

Three strategies are implemented behind one interface, because Section 5 calls
for an ablation that benchmarks the learned weighting against simpler
baselines. Having them share a signature means the ablation is a loop over
strategy names rather than three divergent code paths:

  logistic  — the default. Sigmoid over a weighted sum, matching the form used
              in the original pipeline prototype. Its virtue is that a single
              high-confidence detection saturates the score instead of being
              averaged away by four clean detectors.
  linear    — the PRD's baseline formula, 100 * sum(w_i * s_i * c_i) with
              weights normalised to sum to 1. Hand-tuned comparison point.
  max       — the crudest baseline: the single worst detector wins. Useful in
              the report as a floor that any real aggregator must beat.

Two behaviours matter for correctness:

  * Every detector's contribution is s_i * c_i, so a detector that fires with
    low confidence moves the score less than one that fires confidently.
  * A detector that failed to run (error set) contributes nothing at all, and
    the linear strategy renormalises over the detectors that did run. Without
    that renormalisation, a deployment missing two detectors would look
    systematically safer than it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas import DetectorReport, Severity

# Coefficients on the logit scale. Data leakage carries the heaviest penalty
# per the PRD; the bias holds a fully clean interaction near zero.
#
# These are hand-set defaults standing in for the logistic regression that
# Section 3 calls for. Replacing this dict with coefficients fitted by
# sklearn on labelled data is a drop-in change — nothing else needs to move.
LOGISTIC_WEIGHTS: dict[str, float] = {
    "prompt_injection": 8.0,
    "jailbreak": 8.0,
    "data_leakage": 8.5,
    "hallucination": 6.0,
    "unsafe_output": 7.5,
}
LOGISTIC_BIAS = -4.5

# Relative importance for the linear baseline. Normalised at call time over
# whichever detectors actually reported.
LINEAR_WEIGHTS: dict[str, float] = {
    "prompt_injection": 0.25,
    "jailbreak": 0.25,
    "data_leakage": 0.20,
    "hallucination": 0.15,
    "unsafe_output": 0.15,
}

# Upper bound of each band, walked in order.
SEVERITY_BANDS: list[tuple[float, Severity]] = [
    (20.0, "Low"),
    (50.0, "Medium"),
    (80.0, "High"),
    (100.0, "Critical"),
]


@dataclass(frozen=True)
class ScoreResult:
    risk_score: float
    severity: Severity
    strategy: str
    # Per-detector contribution s_i * c_i, kept for the dashboard's drill-down
    # and for explaining a score during a review.
    contributions: dict[str, float]


def _contributions(reports: list[DetectorReport]) -> dict[str, float]:
    """s_i * c_i per detector, skipping any detector that errored."""
    return {
        r.detector_name: r.sub_score * r.confidence
        for r in reports
        if r.error is None
    }


def severity_for(score: float) -> Severity:
    for upper, band in SEVERITY_BANDS:
        if score <= upper:
            return band
    return "Critical"


def _logistic(contrib: dict[str, float]) -> float:
    logit = LOGISTIC_BIAS + sum(
        LOGISTIC_WEIGHTS.get(name, 0.0) * value for name, value in contrib.items()
    )
    return 100.0 / (1.0 + math.exp(-logit))


def _linear(contrib: dict[str, float]) -> float:
    weights = {n: LINEAR_WEIGHTS.get(n, 0.0) for n in contrib}
    total = sum(weights.values())
    if total == 0:
        return 0.0
    return 100.0 * sum(
        (w / total) * contrib[n] for n, w in weights.items()
    )


def _max(contrib: dict[str, float]) -> float:
    return 100.0 * max(contrib.values(), default=0.0)


_STRATEGIES = {"logistic": _logistic, "linear": _linear, "max": _max}


def score(
    reports: list[DetectorReport], strategy: str = "logistic"
) -> ScoreResult:
    """
    Aggregate detector reports into a composite risk score.

    Raises ValueError on an unknown strategy rather than silently falling back,
    so an ablation typo shows up immediately instead of quietly producing
    logistic numbers under another label.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown scoring strategy {strategy!r}; "
            f"expected one of {sorted(_STRATEGIES)}"
        )

    contrib = _contributions(reports)
    raw = _STRATEGIES[strategy](contrib)
    risk = round(min(100.0, max(0.0, raw)), 1)

    return ScoreResult(
        risk_score=risk,
        severity=severity_for(risk),
        strategy=strategy,
        contributions={k: round(v, 4) for k, v in contrib.items()},
    )
