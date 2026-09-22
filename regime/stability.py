from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any


def _finite_rate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    position: int
    regime: str
    confidence: float

    def __post_init__(self) -> None:
        _non_negative_integer("position", self.position)
        if not isinstance(self.regime, str) or not self.regime.strip():
            raise ValueError("regime must be a non-empty string")
        if self.regime != self.regime.strip():
            raise ValueError("regime must not contain surrounding whitespace")
        _finite_rate("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class RegimeStabilityPolicy:
    min_observations: int = 60
    min_run_length: int = 3
    max_switch_rate: float = 0.15
    max_short_run_fraction: float = 0.20
    max_immediate_reversal_rate: float = 0.50
    min_mean_confidence: float = 0.60
    max_dominant_regime_fraction: float = 0.95

    def __post_init__(self) -> None:
        _non_negative_integer("min_observations", self.min_observations)
        _non_negative_integer("min_run_length", self.min_run_length)
        if self.min_observations == 0:
            raise ValueError("min_observations must be positive")
        if self.min_run_length == 0:
            raise ValueError("min_run_length must be positive")
        _finite_rate("max_switch_rate", self.max_switch_rate)
        _finite_rate("max_short_run_fraction", self.max_short_run_fraction)
        _finite_rate("max_immediate_reversal_rate", self.max_immediate_reversal_rate)
        _finite_rate("min_mean_confidence", self.min_mean_confidence)
        _finite_rate(
            "max_dominant_regime_fraction", self.max_dominant_regime_fraction
        )
        if self.max_dominant_regime_fraction == 0:
            raise ValueError("max_dominant_regime_fraction must be greater than zero")


@dataclass(frozen=True, slots=True)
class RegimeRun:
    regime: str
    start_position: int
    end_position: int
    length: int
    mean_confidence: float


@dataclass(frozen=True, slots=True)
class RegimeStabilityReport:
    passed: bool
    reasons: tuple[str, ...]
    observations: int
    regime_count: int
    run_count: int
    switch_rate: float
    short_run_fraction: float
    immediate_reversal_rate: float
    mean_confidence: float
    dominant_regime: str | None
    dominant_regime_fraction: float
    runs: tuple[RegimeRun, ...]
    per_regime: dict[str, dict[str, float | int]]
    policy: RegimeStabilityPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "summary": {
                "observations": self.observations,
                "regime_count": self.regime_count,
                "run_count": self.run_count,
                "switch_rate": round(self.switch_rate, 6),
                "short_run_fraction": round(self.short_run_fraction, 6),
                "immediate_reversal_rate": round(
                    self.immediate_reversal_rate, 6
                ),
                "mean_confidence": round(self.mean_confidence, 6),
                "dominant_regime": self.dominant_regime,
                "dominant_regime_fraction": round(
                    self.dominant_regime_fraction, 6
                ),
            },
            "per_regime": self.per_regime,
            "runs": [asdict(run) for run in self.runs],
            "policy": asdict(self.policy),
        }


def audit_regime_stability(
    observations: Iterable[RegimeObservation],
    policy: RegimeStabilityPolicy | None = None,
) -> RegimeStabilityReport:
    values = tuple(observations)
    active_policy = policy or RegimeStabilityPolicy()
    for previous, current in pairwise(values):
        if current.position <= previous.position:
            raise ValueError("observation positions must be unique and strictly increasing")

    runs = _build_runs(values)
    observation_count = len(values)
    regime_counts = Counter(item.regime for item in values)
    dominant_regime = None
    dominant_fraction = 0.0
    if regime_counts:
        dominant_regime, dominant_count = min(
            regime_counts.items(), key=lambda item: (-item[1], item[0])
        )
        dominant_fraction = dominant_count / observation_count

    switches = max(len(runs) - 1, 0)
    switch_rate = switches / (observation_count - 1) if observation_count > 1 else 0.0
    short_observations = sum(
        run.length for run in runs if run.length < active_policy.min_run_length
    )
    short_run_fraction = (
        short_observations / observation_count if observation_count else 0.0
    )
    possible_reversals = max(len(runs) - 2, 0)
    reversals = sum(
        previous.regime == following.regime
        and current.length < active_policy.min_run_length
        for previous, current, following in zip(
            runs, runs[1:], runs[2:], strict=False
        )
    )
    reversal_rate = reversals / possible_reversals if possible_reversals else 0.0
    mean_confidence = (
        sum(item.confidence for item in values) / observation_count
        if observation_count
        else 0.0
    )

    reasons: list[str] = []
    if observation_count < active_policy.min_observations:
        reasons.append("insufficient_observations")
    if mean_confidence < active_policy.min_mean_confidence:
        reasons.append("mean_confidence_below_minimum")
    if switch_rate > active_policy.max_switch_rate:
        reasons.append("switch_rate_exceeded")
    if short_run_fraction > active_policy.max_short_run_fraction:
        reasons.append("short_run_fraction_exceeded")
    if reversal_rate > active_policy.max_immediate_reversal_rate:
        reasons.append("immediate_reversal_rate_exceeded")
    if dominant_fraction > active_policy.max_dominant_regime_fraction:
        reasons.append("dominant_regime_fraction_exceeded")

    per_regime: dict[str, dict[str, float | int]] = {}
    for regime in sorted(regime_counts):
        regime_runs = [run for run in runs if run.regime == regime]
        lengths = [run.length for run in regime_runs]
        per_regime[regime] = {
            "observations": regime_counts[regime],
            "runs": len(regime_runs),
            "mean_run_length": round(sum(lengths) / len(lengths), 6),
            "min_run_length": min(lengths),
            "max_run_length": max(lengths),
        }

    return RegimeStabilityReport(
        passed=not reasons,
        reasons=tuple(reasons),
        observations=observation_count,
        regime_count=len(regime_counts),
        run_count=len(runs),
        switch_rate=switch_rate,
        short_run_fraction=short_run_fraction,
        immediate_reversal_rate=reversal_rate,
        mean_confidence=mean_confidence,
        dominant_regime=dominant_regime,
        dominant_regime_fraction=dominant_fraction,
        runs=runs,
        per_regime=per_regime,
        policy=active_policy,
    )


def _build_runs(values: tuple[RegimeObservation, ...]) -> tuple[RegimeRun, ...]:
    if not values:
        return ()
    runs: list[RegimeRun] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and values[index].regime == values[start].regime:
            continue
        run_values = values[start:index]
        runs.append(
            RegimeRun(
                regime=run_values[0].regime,
                start_position=run_values[0].position,
                end_position=run_values[-1].position,
                length=len(run_values),
                mean_confidence=round(
                    sum(item.confidence for item in run_values) / len(run_values), 6
                ),
            )
        )
        start = index
    return tuple(runs)
