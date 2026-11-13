from __future__ import annotations

import json

import pytest

from regime.occupancy_drift import (
    OccupancyDriftError,
    OccupancyDriftPolicy,
    audit_occupancy_drift,
    load_evidence,
    main,
)


def _evidence(
    *,
    reference: dict[str, int] | None = None,
    current: dict[str, int] | None = None,
    reference_effective: int | None = None,
    current_effective: int | None = None,
) -> dict[str, object]:
    reference_window: dict[str, object] = {
        "counts": reference or {"calm": 500, "transition": 300, "stress": 200}
    }
    current_window: dict[str, object] = {
        "counts": current or {"calm": 100, "transition": 60, "stress": 40}
    }
    if reference_effective is not None:
        reference_window["effective_sample_size"] = reference_effective
    if current_effective is not None:
        current_window["effective_sample_size"] = current_effective
    return {
        "schema_version": 1,
        "governed_states": ["stress", "calm", "transition"],
        "reference": reference_window,
        "current": current_window,
    }


def test_matching_occupancy_passes_and_is_canonical():
    first = audit_occupancy_drift(_evidence())
    reordered = _evidence(
        reference={"stress": 200, "calm": 500, "transition": 300},
        current={"stress": 40, "calm": 100, "transition": 60},
    )
    reordered["governed_states"] = ["transition", "stress", "calm"]
    second = audit_occupancy_drift(reordered)

    assert first["passed"] is True
    assert first["summary"]["js_divergence"] == pytest.approx(0.0, abs=1e-5)
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["decision_sha256"] == second["decision_sha256"]


def test_material_distribution_shift_fails_effect_size_gates():
    report = audit_occupancy_drift(
        _evidence(current={"calm": 20, "transition": 30, "stress": 150})
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["passed"] is False
    assert "js_divergence_exceeded" in codes
    assert "state_share_shift_exceeded" in codes
    assert report["summary"]["statistically_resolved_state_count"] >= 1


def test_emergent_state_has_a_separate_finding():
    report = audit_occupancy_drift(
        _evidence(
            reference={"calm": 700, "transition": 300, "stress": 0},
            current={"calm": 100, "transition": 60, "stress": 40},
        )
    )

    emergent = [
        finding
        for finding in report["findings"]
        if finding["code"] == "emergent_state_share_exceeded"
    ]
    assert emergent == [
        {
            "code": "emergent_state_share_exceeded",
            "state": "stress",
            "detail": "current_share=0.200000; maximum=0.050000",
        }
    ]


def test_effective_sample_size_changes_uncertainty_not_effect_size():
    full = audit_occupancy_drift(_evidence())
    dependent = audit_occupancy_drift(
        _evidence(reference_effective=100, current_effective=20)
    )

    assert dependent["summary"]["js_divergence"] == full["summary"]["js_divergence"]
    assert (
        dependent["summary"]["simultaneous_sampling_bound"]
        > full["summary"]["simultaneous_sampling_bound"]
    )


def test_insufficient_windows_fail_closed():
    report = audit_occupancy_drift(
        _evidence(
            reference={"calm": 50, "transition": 30, "stress": 20},
            current={"calm": 10, "transition": 6, "stress": 4},
        )
    )

    assert {finding["code"] for finding in report["findings"]} == {
        "insufficient_current_observations",
        "insufficient_reference_observations",
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda item: item["current"]["counts"].pop("stress"),
            "exactly the governed states",
        ),
        (
            lambda item: item["current"]["counts"].update(stress=True),
            "non-negative integer",
        ),
        (
            lambda item: item["governed_states"].append("calm"),
            "must not contain duplicates",
        ),
        (
            lambda item: item["current"].update(effective_sample_size=201),
            "between 1 and total",
        ),
    ],
)
def test_malformed_evidence_is_rejected(mutate, match):
    evidence = _evidence()
    mutate(evidence)
    with pytest.raises(OccupancyDriftError, match=match):
        audit_occupancy_drift(evidence)


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(OccupancyDriftError, match="duplicate JSON key"):
        load_evidence(path)


def test_cli_writes_report_and_returns_policy_rejection(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "report.json"
    evidence_path.write_text(
        json.dumps(_evidence(current={"calm": 20, "transition": 30, "stress": 150})),
        encoding="utf-8",
    )

    exit_code = main([str(evidence_path), "--output", str(output_path)])

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert len(report["decision_sha256"]) == 64


def test_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="max_js_divergence"):
        OccupancyDriftPolicy(max_js_divergence=0.0)
