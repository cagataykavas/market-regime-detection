from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from regime.transition_uncertainty import (
    INVALID_INPUT_EXIT_CODE,
    PASS_EXIT_CODE,
    POLICY_FAILURE_EXIT_CODE,
    RegimePoint,
    TransitionUncertaintyPolicy,
    audit_transition_uncertainty,
    main,
)


def alternating_points(count: int) -> list[RegimePoint]:
    return [
        RegimePoint(position=index, regime="calm" if index % 2 == 0 else "stress")
        for index in range(count)
    ]


def test_well_supported_transitions_pass_with_simultaneous_bounds() -> None:
    report = audit_transition_uncertainty(alternating_points(121))

    assert report.passed
    assert report.origin_totals == {"calm": 60, "stress": 60}
    assert len(report.estimates) == 4
    assert report.per_interval_confidence > report.simultaneous_confidence
    assert all(estimate.interval_width < 0.20 for estimate in report.estimates)
    assert len(report.evidence_digest) == 64
    assert report.as_dict()["schema_version"] == "regime-transition-uncertainty-v1"


def test_insufficient_transition_support_fails_policy() -> None:
    report = audit_transition_uncertainty(
        alternating_points(20),
        policy=TransitionUncertaintyPolicy(
            min_observations=20,
            min_outgoing_transitions=15,
        ),
    )

    assert not report.passed
    assert {violation.code for violation in report.violations} == {
        "insufficient_outgoing_transitions"
    }
    assert {violation.origin for violation in report.violations} == {"calm", "stress"}


def test_wide_intervals_fail_even_when_minimum_counts_pass() -> None:
    report = audit_transition_uncertainty(
        alternating_points(21),
        policy=TransitionUncertaintyPolicy(
            min_observations=20,
            min_outgoing_transitions=10,
            max_interval_width=0.10,
        ),
    )

    assert not report.passed
    assert {violation.code for violation in report.violations} == {
        "interval_width_exceeded"
    }


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (
            [
                RegimePoint(index, "calm" if index % 2 == 0 else "stress")
                for index in range(60)
            ]
            + [RegimePoint(61, "calm")],
            "contiguous",
        ),
        ([RegimePoint(index, "single") for index in range(60)], "regime count"),
    ],
)
def test_invalid_sequences_fail_closed(points: list[RegimePoint], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_transition_uncertainty(points)


def test_cli_uses_distinct_pass_policy_and_invalid_exit_codes(tmp_path: Path) -> None:
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    invalid = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    passing.write_text(
        json.dumps([asdict(point) for point in alternating_points(121)]),
        encoding="utf-8",
    )
    failing.write_text(
        json.dumps([asdict(point) for point in alternating_points(20)]),
        encoding="utf-8",
    )
    invalid.write_text("{}", encoding="utf-8")

    assert main([str(passing), "--output", str(output)]) == PASS_EXIT_CODE
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert (
        main(
            [
                str(failing),
                "--min-observations",
                "20",
                "--min-outgoing-transitions",
                "15",
            ]
        )
        == POLICY_FAILURE_EXIT_CODE
    )
    assert main([str(invalid)]) == INVALID_INPUT_EXIT_CODE
