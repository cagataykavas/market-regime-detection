from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 1024 * 1024
MAX_OBSERVATIONS = 10_000_000


class OccupancyDriftError(ValueError):
    """Raised when occupancy evidence is malformed or internally inconsistent."""


@dataclass(frozen=True)
class OccupancyDriftPolicy:
    min_reference_observations: int = 250
    min_current_observations: int = 60
    max_js_divergence: float = 0.05
    max_abs_state_shift: float = 0.15
    max_emergent_state_share: float = 0.05
    familywise_alpha: float = 0.05
    pseudocount: float = 0.5

    def __post_init__(self) -> None:
        if self.min_reference_observations < 1 or self.min_current_observations < 1:
            raise ValueError("minimum observation counts must be positive")
        for name in (
            "max_js_divergence",
            "max_abs_state_shift",
            "max_emergent_state_share",
            "familywise_alpha",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if not math.isfinite(self.pseudocount) or not 0.0 < self.pseudocount <= 1.0:
            raise ValueError("pseudocount must be finite and in (0, 1]")


@dataclass(frozen=True, order=True)
class DriftFinding:
    code: str
    state: str | None
    detail: str


@dataclass(frozen=True)
class _Window:
    counts: dict[str, int]
    total: int
    effective_sample_size: int


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OccupancyDriftError(f"{context} must be an object")
    return value


def _governed_states(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not 2 <= len(value) <= 12:
        raise OccupancyDriftError(
            "governed_states must contain between 2 and 12 states"
        )
    if not all(
        isinstance(state, str)
        and state == state.strip()
        and bool(state)
        and len(state) <= 64
        for state in value
    ):
        raise OccupancyDriftError(
            "governed_states must contain non-empty strings of at most 64 characters"
        )
    if len(set(value)) != len(value):
        raise OccupancyDriftError("governed_states must not contain duplicates")
    return tuple(sorted(value))


def _window(value: Any, states: tuple[str, ...], context: str) -> _Window:
    raw_window = _require_mapping(value, context)
    raw_counts = _require_mapping(raw_window.get("counts"), f"{context}.counts")
    if set(raw_counts) != set(states):
        missing = len(set(states) - set(raw_counts))
        unknown = len(set(raw_counts) - set(states))
        raise OccupancyDriftError(
            f"{context}.counts must contain exactly the governed states; missing={missing}, unknown={unknown}"
        )
    counts: dict[str, int] = {}
    for state in states:
        count = raw_counts[state]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise OccupancyDriftError(
                f"{context}.counts[{state!r}] must be a non-negative integer"
            )
        counts[state] = count
    total = sum(counts.values())
    if total < 1:
        raise OccupancyDriftError(f"{context} must contain at least one observation")
    if total > MAX_OBSERVATIONS:
        raise OccupancyDriftError(f"{context} exceeds {MAX_OBSERVATIONS} observations")
    effective = raw_window.get("effective_sample_size", total)
    if (
        isinstance(effective, bool)
        or not isinstance(effective, int)
        or not 1 <= effective <= total
    ):
        raise OccupancyDriftError(
            f"{context}.effective_sample_size must be an integer between 1 and total observations"
        )
    return _Window(counts=counts, total=total, effective_sample_size=effective)


def _smoothed_proportions(
    window: _Window, states: tuple[str, ...], alpha: float
) -> dict[str, float]:
    denominator = window.total + alpha * len(states)
    return {state: (window.counts[state] + alpha) / denominator for state in states}


def _raw_proportions(window: _Window, states: tuple[str, ...]) -> dict[str, float]:
    return {state: window.counts[state] / window.total for state in states}


def _js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    total = 0.0
    for state in sorted(left):
        midpoint = 0.5 * (left[state] + right[state])
        total += 0.5 * left[state] * math.log2(left[state] / midpoint)
        total += 0.5 * right[state] * math.log2(right[state] / midpoint)
    return total


def _normalized_entropy(proportions: Mapping[str, float]) -> float:
    return -math.fsum(
        value * math.log2(value) for value in proportions.values()
    ) / math.log2(len(proportions))


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_occupancy_drift(
    evidence: Mapping[str, Any],
    policy: OccupancyDriftPolicy | None = None,
) -> dict[str, Any]:
    """Compare reference and current regime occupancy using count-only evidence."""
    policy = policy or OccupancyDriftPolicy()
    if evidence.get("schema_version") != 1:
        raise OccupancyDriftError("schema_version must equal 1")
    states = _governed_states(evidence.get("governed_states"))
    reference = _window(evidence.get("reference"), states, "reference")
    current = _window(evidence.get("current"), states, "current")

    reference_raw = _raw_proportions(reference, states)
    current_raw = _raw_proportions(current, states)
    reference_smoothed = _smoothed_proportions(reference, states, policy.pseudocount)
    current_smoothed = _smoothed_proportions(current, states, policy.pseudocount)
    divergence = _js_divergence(reference_smoothed, current_smoothed)
    simultaneous_bound = math.sqrt(
        math.log(2.0 * len(states) / policy.familywise_alpha)
        / (2.0 * reference.effective_sample_size)
    ) + math.sqrt(
        math.log(2.0 * len(states) / policy.familywise_alpha)
        / (2.0 * current.effective_sample_size)
    )

    findings: list[DriftFinding] = []
    if reference.total < policy.min_reference_observations:
        findings.append(
            DriftFinding(
                "insufficient_reference_observations",
                None,
                f"found={reference.total}; minimum={policy.min_reference_observations}",
            )
        )
    if current.total < policy.min_current_observations:
        findings.append(
            DriftFinding(
                "insufficient_current_observations",
                None,
                f"found={current.total}; minimum={policy.min_current_observations}",
            )
        )
    if divergence > policy.max_js_divergence:
        findings.append(
            DriftFinding(
                "js_divergence_exceeded",
                None,
                f"observed={divergence:.6f}; maximum={policy.max_js_divergence:.6f}",
            )
        )

    state_rows: list[dict[str, Any]] = []
    for state in states:
        shift = current_raw[state] - reference_raw[state]
        absolute_shift = abs(shift)
        if absolute_shift > policy.max_abs_state_shift:
            findings.append(
                DriftFinding(
                    "state_share_shift_exceeded",
                    state,
                    f"absolute_shift={absolute_shift:.6f}; maximum={policy.max_abs_state_shift:.6f}",
                )
            )
        if (
            reference.counts[state] == 0
            and current_raw[state] > policy.max_emergent_state_share
        ):
            findings.append(
                DriftFinding(
                    "emergent_state_share_exceeded",
                    state,
                    f"current_share={current_raw[state]:.6f}; maximum={policy.max_emergent_state_share:.6f}",
                )
            )
        state_rows.append(
            {
                "state": state,
                "reference_count": reference.counts[state],
                "current_count": current.counts[state],
                "reference_share": reference_raw[state],
                "current_share": current_raw[state],
                "share_shift": shift,
                "statistically_resolved": absolute_shift > simultaneous_bound,
            }
        )

    findings.sort()
    normalized_evidence = {
        "schema_version": 1,
        "governed_states": list(states),
        "reference": asdict(reference),
        "current": asdict(current),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": not findings,
        "evidence_sha256": _canonical_digest(normalized_evidence),
        "policy": asdict(policy),
        "summary": {
            "reference_observations": reference.total,
            "current_observations": current.total,
            "reference_effective_sample_size": reference.effective_sample_size,
            "current_effective_sample_size": current.effective_sample_size,
            "js_divergence": divergence,
            "max_abs_state_shift": max(abs(row["share_shift"]) for row in state_rows),
            "simultaneous_sampling_bound": simultaneous_bound,
            "statistically_resolved_state_count": sum(
                row["statistically_resolved"] for row in state_rows
            ),
            "reference_normalized_entropy": _normalized_entropy(reference_smoothed),
            "current_normalized_entropy": _normalized_entropy(current_smoothed),
            "finding_count": len(findings),
        },
        "states": state_rows,
        "findings": [asdict(finding) for finding in findings],
    }
    report["decision_sha256"] = _canonical_digest(report)
    return report


def _reject_constant(value: str) -> None:
    raise OccupancyDriftError(f"non-finite JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OccupancyDriftError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_evidence(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OccupancyDriftError("input must be a regular, non-symlink file")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise OccupancyDriftError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OccupancyDriftError(f"could not read valid JSON: {exc}") from exc
    return _require_mapping(parsed, "evidence")


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit named market-regime occupancy drift"
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/occupancy-drift.json")
    )
    parser.add_argument("--min-reference", type=int, default=250)
    parser.add_argument("--min-current", type=int, default=60)
    parser.add_argument("--max-js", type=float, default=0.05)
    parser.add_argument("--max-share-shift", type=float, default=0.15)
    parser.add_argument("--max-emergent-share", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        policy = OccupancyDriftPolicy(
            min_reference_observations=args.min_reference,
            min_current_observations=args.min_current,
            max_js_divergence=args.max_js,
            max_abs_state_shift=args.max_share_shift,
            max_emergent_state_share=args.max_emergent_share,
        )
        report = audit_occupancy_drift(load_evidence(args.evidence), policy)
        write_report(report, args.output)
    except (OccupancyDriftError, OSError, ValueError) as exc:
        parser.exit(2, f"occupancy drift audit error: {exc}\n")
    print(
        json.dumps(
            {"passed": report["passed"], **report["summary"]}, indent=2, sort_keys=True
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
