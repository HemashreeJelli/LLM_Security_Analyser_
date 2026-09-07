"""Risk-scoring engine behaviour, including the degraded-deployment path."""

from __future__ import annotations

import pytest

from app.schemas import DetectorReport
from app.scoring import score, severity_for


def report(name: str, sub: float, conf: float, error: str | None = None) -> DetectorReport:
    return DetectorReport(
        detector_name=name,
        is_flagged=sub > 0.5,
        confidence=conf,
        sub_score=sub,
        evidence={},
        error=error,
    )


def test_clean_traffic_scores_low_on_every_strategy():
    reports = [
        report("prompt_injection", 0.02, 0.98),
        report("jailbreak", 0.01, 0.99),
        report("data_leakage", 0.0, 1.0),
    ]
    for strategy in ("logistic", "linear", "max"):
        result = score(reports, strategy)
        assert result.severity == "Low", strategy
        assert result.risk_score < 20.0


def test_confident_attack_saturates_the_logistic_score():
    reports = [
        report("prompt_injection", 0.95, 0.93),
        report("jailbreak", 0.88, 0.90),
        report("data_leakage", 0.0, 1.0),
    ]
    result = score(reports, "logistic")
    assert result.severity == "Critical"
    assert result.risk_score > 95.0


def test_low_confidence_detection_is_discounted():
    """A detector firing at low confidence must move the score less than a
    confident one, which is the whole reason confidence is a separate field."""
    confident = score([report("jailbreak", 0.9, 0.95)], "logistic")
    unsure = score([report("jailbreak", 0.9, 0.15)], "logistic")
    assert unsure.risk_score < confident.risk_score


def test_errored_detector_contributes_nothing():
    with_error = score(
        [
            report("prompt_injection", 0.99, 0.99, error="weights missing"),
            report("data_leakage", 0.0, 1.0),
        ]
    )
    assert with_error.risk_score < 20.0
    assert "prompt_injection" not in with_error.contributions


def test_linear_renormalises_over_detectors_that_ran():
    """Two detectors present must not look safer than five simply because the
    missing weights would otherwise dilute the sum."""
    partial = score([report("data_leakage", 1.0, 1.0)], "linear")
    assert partial.risk_score == pytest.approx(100.0)


def test_no_detectors_at_all_is_not_a_clean_verdict_by_accident():
    result = score([], "linear")
    assert result.risk_score == 0.0
    assert result.contributions == {}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown scoring strategy"):
        score([report("jailbreak", 0.5, 0.5)], "logisitc")


@pytest.mark.parametrize(
    ("value", "band"),
    [(0.0, "Low"), (20.0, "Low"), (20.1, "Medium"), (50.0, "Medium"),
     (50.1, "High"), (80.0, "High"), (80.1, "Critical"), (100.0, "Critical")],
)
def test_severity_band_boundaries(value: float, band: str):
    assert severity_for(value) == band
