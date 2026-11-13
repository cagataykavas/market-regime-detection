from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist

PASS_EXIT_CODE = 0
INVALID_INPUT_EXIT_CODE = 2
POLICY_FAILURE_EXIT_CODE = 3
MAX_INPUT_BYTES = 16_000_000
MAX_REGIMES_LIMIT = 32
MAX_OBSERVATIONS_LIMIT = 100_000


def _integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _rate(name: str, value: float, *, inclusive_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    valid_lower_bound = value >= 0.0 if inclusive_zero else value > 0.0
    if not math.isfinite(value) or not valid_lower_bound or value > 1.0:
        qualifier = (
            "between 0 and 1" if inclusive_zero else "greater than 0 and at most 1"
        )
        raise ValueError(f"{name} must be finite and {qualifier}")


@dataclass(frozen=True, slots=True)
class RegimePoint:
    position: int
    regime: str

    def __post_init__(self) -> None:
        _integer("position", self.position, minimum=0)
        if not isinstance(self.regime, str) or not self.regime.strip():
            raise ValueError("regime must be a non-empty string")
        if self.regime != self.regime.strip() or len(self.regime) > 100:
            raise ValueError("regime must be trimmed and at most 100 characters")
        if any(ord(character) < 32 for character in self.regime):
            raise ValueError("regime must not contain control characters")


@dataclass(frozen=True, slots=True)
class TransitionUncertaintyPolicy:
    confidence: float = 0.95
    min_observations: int = 60
    min_outgoing_transitions: int = 30
    max_interval_width: float = 0.50
    max_regimes: int = 8
    max_observations: int = MAX_OBSERVATIONS_LIMIT

    def __post_init__(self) -> None:
        _rate("confidence", self.confidence)
        if self.confidence >= 1.0:
            raise ValueError("confidence must be less than 1")
        _rate("max_interval_width", self.max_interval_width)
        _integer("min_observations", self.min_observations, minimum=2)
        _integer("min_outgoing_transitions", self.min_outgoing_transitions, minimum=1)
        _integer("max_regimes", self.max_regimes, minimum=2)
        _integer(
            "max_observations", self.max_observations, minimum=self.min_observations
        )
        if self.max_regimes > MAX_REGIMES_LIMIT:
            raise ValueError(f"max_regimes must not exceed {MAX_REGIMES_LIMIT}")
        if self.max_observations > MAX_OBSERVATIONS_LIMIT:
            raise ValueError(
                f"max_observations must not exceed {MAX_OBSERVATIONS_LIMIT}"
            )


@dataclass(frozen=True, slots=True)
class TransitionEstimate:
    origin: str
    destination: str
    transitions: int
    origin_transitions: int
    probability: float
    lower_bound: float
    upper_bound: float
    interval_width: float


@dataclass(frozen=True, slots=True)
class TransitionViolation:
    code: str
    origin: str | None = None
    destination: str | None = None


@dataclass(frozen=True, slots=True)
class TransitionUncertaintyReport:
    passed: bool
    observations: int
    regimes: tuple[str, ...]
    simultaneous_confidence: float
    per_interval_confidence: float
    origin_totals: dict[str, int]
    estimates: tuple[TransitionEstimate, ...]
    violations: tuple[TransitionViolation, ...]
    evidence_digest: str
    policy: TransitionUncertaintyPolicy

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "regime-transition-uncertainty-v1"
        return payload


def _wilson_interval(
    successes: int, trials: int, z_score: float
) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    probability = successes / trials
    z_squared = z_score**2
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / trials + z_squared / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _evidence_digest(
    points: Sequence[RegimePoint], policy: TransitionUncertaintyPolicy
) -> str:
    payload = {
        "schema_version": "regime-transition-evidence-v1",
        "points": [asdict(point) for point in points],
        "policy": asdict(policy),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_transition_uncertainty(
    points: Sequence[RegimePoint],
    *,
    policy: TransitionUncertaintyPolicy | None = None,
) -> TransitionUncertaintyReport:
    active_policy = policy or TransitionUncertaintyPolicy()
    values = tuple(points)
    if (
        not active_policy.min_observations
        <= len(values)
        <= active_policy.max_observations
    ):
        raise ValueError(
            f"observation count must be between {active_policy.min_observations} "
            f"and {active_policy.max_observations}"
        )
    for previous, current in pairwise(values):
        if current.position != previous.position + 1:
            raise ValueError(
                "observation positions must be unique, ordered, and contiguous"
            )

    regimes = tuple(sorted({point.regime for point in values}))
    if not 2 <= len(regimes) <= active_policy.max_regimes:
        raise ValueError(
            f"regime count must be between 2 and {active_policy.max_regimes}"
        )

    counts: Counter[tuple[str, str]] = Counter(
        (current.regime, following.regime) for current, following in pairwise(values)
    )
    origin_totals = {
        origin: sum(counts[(origin, destination)] for destination in regimes)
        for origin in regimes
    }
    hypotheses = len(regimes) ** 2
    family_alpha = 1.0 - active_policy.confidence
    per_interval_alpha = family_alpha / hypotheses
    z_score = NormalDist().inv_cdf(1.0 - per_interval_alpha / 2.0)
    per_interval_confidence = 1.0 - per_interval_alpha

    estimates: list[TransitionEstimate] = []
    violations: list[TransitionViolation] = []
    for origin in regimes:
        trials = origin_totals[origin]
        if trials < active_policy.min_outgoing_transitions:
            violations.append(
                TransitionViolation("insufficient_outgoing_transitions", origin=origin)
            )
        for destination in regimes:
            transitions = counts[(origin, destination)]
            lower, upper = _wilson_interval(transitions, trials, z_score)
            width = upper - lower
            estimates.append(
                TransitionEstimate(
                    origin=origin,
                    destination=destination,
                    transitions=transitions,
                    origin_transitions=trials,
                    probability=transitions / trials if trials else 0.0,
                    lower_bound=lower,
                    upper_bound=upper,
                    interval_width=width,
                )
            )
            if (
                trials >= active_policy.min_outgoing_transitions
                and width > active_policy.max_interval_width
            ):
                violations.append(
                    TransitionViolation(
                        "interval_width_exceeded",
                        origin=origin,
                        destination=destination,
                    )
                )

    return TransitionUncertaintyReport(
        passed=not violations,
        observations=len(values),
        regimes=regimes,
        simultaneous_confidence=active_policy.confidence,
        per_interval_confidence=per_interval_confidence,
        origin_totals=origin_totals,
        estimates=tuple(estimates),
        violations=tuple(violations),
        evidence_digest=_evidence_digest(values, active_policy),
        policy=active_policy,
    )


def _load_points(path: Path) -> list[RegimePoint]:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("input must be a JSON array")
    points: list[RegimePoint] = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"position", "regime"}:
            raise ValueError("each row must contain exactly position and regime")
        points.append(RegimePoint(position=row["position"], regime=row["regime"]))
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit uncertainty in empirical regime transition probabilities."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--min-outgoing-transitions", type=int, default=30)
    parser.add_argument("--max-interval-width", type=float, default=0.50)
    parser.add_argument("--max-regimes", type=int, default=8)
    parser.add_argument("--max-observations", type=int, default=MAX_OBSERVATIONS_LIMIT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        policy = TransitionUncertaintyPolicy(
            confidence=arguments.confidence,
            min_observations=arguments.min_observations,
            min_outgoing_transitions=arguments.min_outgoing_transitions,
            max_interval_width=arguments.max_interval_width,
            max_regimes=arguments.max_regimes,
            max_observations=arguments.max_observations,
        )
        report = audit_transition_uncertainty(
            _load_points(arguments.input), policy=policy
        )
        rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.output is None:
            print(rendered, end="")
        else:
            arguments.output.write_text(rendered, encoding="utf-8")
    except (OSError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"transition uncertainty audit failed: {exc}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE
    return PASS_EXIT_CODE if report.passed else POLICY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
