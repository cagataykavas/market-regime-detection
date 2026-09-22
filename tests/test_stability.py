from __future__ import annotations

import json

import pytest
from regime.stability import (
    RegimeObservation,
    RegimeStabilityPolicy,
    audit_regime_stability,
)


def observations(regimes: list[str], confidence: float = 0.9):
    return [
        RegimeObservation(position=index, regime=regime, confidence=confidence)
        for index, regime in enumerate(regimes)
    ]


def test_persistent_regimes_pass_and_report_run_evidence() -> None:
    values = observations(["calm"] * 30 + ["transition"] * 10 + ["stress"] * 30)

    report = audit_regime_stability(values)

    assert report.passed is True
    assert report.run_count == 3
    assert report.switch_rate == pytest.approx(2 / 69)
    assert report.short_run_fraction == 0.0
    assert report.immediate_reversal_rate == 0.0
    assert report.dominant_regime == "calm"
    assert report.runs[1].length == 10
    assert report.per_regime["stress"]["mean_run_length"] == 30.0


def test_churn_reversals_and_low_confidence_fail_together() -> None:
    values = observations(["calm", "stress"] * 35, confidence=0.4)

    report = audit_regime_stability(values)

    assert report.passed is False
    assert report.reasons == (
        "mean_confidence_below_minimum",
        "switch_rate_exceeded",
        "short_run_fraction_exceeded",
        "immediate_reversal_rate_exceeded",
    )
    assert report.short_run_fraction == 1.0
    assert report.immediate_reversal_rate == 1.0


def test_collapsed_regime_and_small_sample_are_rejected() -> None:
    report = audit_regime_stability(observations(["calm"] * 20))

    assert report.reasons == (
        "insufficient_observations",
        "dominant_regime_fraction_exceeded",
    )


def test_short_runs_measure_observations_not_only_run_count() -> None:
    values = observations(["calm"] * 8 + ["stress"] + ["calm"] * 11)
    policy = RegimeStabilityPolicy(
        min_observations=20,
        max_short_run_fraction=0.04,
        max_immediate_reversal_rate=1.0,
        max_dominant_regime_fraction=1.0,
    )

    report = audit_regime_stability(values, policy)

    assert report.short_run_fraction == 0.05
    assert report.reasons == ("short_run_fraction_exceeded",)


def test_report_is_json_ready_and_regime_order_is_stable() -> None:
    policy = RegimeStabilityPolicy(
        min_observations=2,
        min_run_length=1,
        max_dominant_regime_fraction=1.0,
    )
    report = audit_regime_stability(observations(["stress", "calm"]), policy)
    payload = json.loads(json.dumps(report.to_dict()))

    assert list(payload["per_regime"]) == ["calm", "stress"]
    assert payload["summary"]["observations"] == 2


@pytest.mark.parametrize(
    "values",
    [
        {"position": -1, "regime": "calm", "confidence": 0.8},
        {"position": True, "regime": "calm", "confidence": 0.8},
        {"position": 0, "regime": "", "confidence": 0.8},
        {"position": 0, "regime": " calm", "confidence": 0.8},
        {"position": 0, "regime": "calm", "confidence": float("nan")},
        {"position": 0, "regime": "calm", "confidence": 1.1},
    ],
)
def test_malformed_observations_fail_closed(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        RegimeObservation(**values)


def test_duplicate_or_unsorted_positions_fail_closed() -> None:
    values = [
        RegimeObservation(1, "calm", 0.8),
        RegimeObservation(1, "stress", 0.8),
    ]

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        audit_regime_stability(values)


@pytest.mark.parametrize(
    "values",
    [
        {"min_observations": 0},
        {"min_run_length": 0},
        {"max_switch_rate": float("inf")},
        {"max_short_run_fraction": -0.1},
        {"max_immediate_reversal_rate": 1.1},
        {"min_mean_confidence": True},
        {"max_dominant_regime_fraction": 0},
    ],
)
def test_invalid_policies_fail_closed(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        RegimeStabilityPolicy(**values)
