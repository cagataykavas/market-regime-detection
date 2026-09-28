from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .engine import FEATURE_COLUMNS, RegimeExperiment, regime_features

MAX_FEATURE_ROWS = 100_000
MAX_REFITS = 256


@dataclass(frozen=True)
class WalkForwardPolicy:
    n_regimes: int = 3
    seed: int = 42
    min_train_size: int = 120
    refit_interval: int = 20
    train_window: int | None = None
    max_feature_rows: int = MAX_FEATURE_ROWS
    max_refits: int = MAX_REFITS

    def __post_init__(self) -> None:
        integer_fields = {
            "n_regimes": self.n_regimes,
            "seed": self.seed,
            "min_train_size": self.min_train_size,
            "refit_interval": self.refit_interval,
            "max_feature_rows": self.max_feature_rows,
            "max_refits": self.max_refits,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise TypeError("walk-forward policy fields must be integers")
        if self.n_regimes not in {2, 3}:
            raise ValueError("n_regimes must be 2 or 3 so semantic labels remain unique")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must fit the unsigned 32-bit range")
        if self.min_train_size < max(30, self.n_regimes * 10):
            raise ValueError("min_train_size is too small for a stable regime fit")
        if self.refit_interval < 1:
            raise ValueError("refit_interval must be positive")
        if self.train_window is not None:
            if isinstance(self.train_window, bool) or not isinstance(self.train_window, int):
                raise TypeError("train_window must be an integer or None")
            if self.train_window < self.min_train_size:
                raise ValueError("train_window cannot be smaller than min_train_size")
        if not self.min_train_size < self.max_feature_rows <= MAX_FEATURE_ROWS:
            raise ValueError(f"max_feature_rows must not exceed {MAX_FEATURE_ROWS}")
        if not 1 <= self.max_refits <= MAX_REFITS:
            raise ValueError(f"max_refits must be between 1 and {MAX_REFITS}")


@dataclass(frozen=True)
class RefitEvidence:
    refit_id: int
    training_start: str
    training_end: str
    prediction_start: str
    prediction_end: str
    training_rows: int
    prediction_rows: int
    regime_names: tuple[str, ...]


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    refits: tuple[RefitEvidence, ...]
    policy: WalkForwardPolicy
    feature_sha256: str

    def evidence(self) -> dict[str, Any]:
        regime_counts = self.predictions["regime"].value_counts().sort_index()
        return {
            "schema": "market-regime-walk-forward/v1",
            "causal_contract": "each prediction uses a fit ending strictly before its timestamp",
            "policy": asdict(self.policy),
            "feature_sha256": self.feature_sha256,
            "prediction_sha256": _prediction_digest(self.predictions),
            "feature_rows": self.policy.min_train_size + len(self.predictions),
            "prediction_rows": len(self.predictions),
            "refit_count": len(self.refits),
            "first_prediction": str(self.predictions.index[0]),
            "last_prediction": str(self.predictions.index[-1]),
            "mean_confidence": float(self.predictions["confidence"].mean()),
            "minimum_confidence": float(self.predictions["confidence"].min()),
            "regime_counts": {str(name): int(count) for name, count in regime_counts.items()},
            "refits": [asdict(refit) for refit in self.refits],
        }


def _validate_features(features: pd.DataFrame, policy: WalkForwardPolicy) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError(f"features must contain exactly {FEATURE_COLUMNS}")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if features.index.tz is None:
        raise ValueError("feature timestamps must be timezone-aware")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("feature timestamps must be strictly increasing and unique")
    if not policy.min_train_size < len(features) <= policy.max_feature_rows:
        raise ValueError(
            f"feature row count must be between {policy.min_train_size + 1} "
            f"and {policy.max_feature_rows}"
        )

    try:
        numeric = features.loc[:, FEATURE_COLUMNS].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("features must contain only finite values")
    return numeric.copy()


def _feature_digest(features: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(b"market-regime-features/v1\0")
    for timestamp, row in features.iterrows():
        digest.update(str(timestamp).encode("utf-8"))
        digest.update(b"\0")
        for value in row:
            digest.update(float(value).hex().encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _prediction_digest(predictions: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(b"market-regime-predictions/v1\0")
    for timestamp, row in predictions.iterrows():
        values = (
            str(timestamp),
            str(row["regime"]),
            str(int(row["component"])),
            float(row["confidence"]).hex(),
            str(int(row["refit_id"])),
            str(row["fit_end"]),
        )
        digest.update("\0".join(values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def classify_walk_forward(
    features: pd.DataFrame,
    policy: WalkForwardPolicy | None = None,
) -> WalkForwardResult:
    """Classify regimes with point-in-time fits and deterministic refit boundaries."""
    resolved_policy = policy or WalkForwardPolicy()
    frame = _validate_features(features, resolved_policy)
    predicted_blocks: list[pd.DataFrame] = []
    refits: list[RefitEvidence] = []
    cursor = resolved_policy.min_train_size

    while cursor < len(frame):
        if len(refits) >= resolved_policy.max_refits:
            raise ValueError("walk-forward evaluation exceeds the configured refit budget")
        training_start = 0
        if resolved_policy.train_window is not None:
            training_start = max(0, cursor - resolved_policy.train_window)
        training = frame.iloc[training_start:cursor]
        prediction_end = min(cursor + resolved_policy.refit_interval, len(frame))
        prediction_features = frame.iloc[cursor:prediction_end]

        model = RegimeExperiment(
            n_regimes=resolved_policy.n_regimes,
            seed=resolved_policy.seed,
        ).fit(training)
        regime_names = tuple(sorted(set(model.component_names.values())))
        if (
            len(model.component_names) != resolved_policy.n_regimes
            or len(regime_names) != resolved_policy.n_regimes
        ):
            raise ValueError("semantic regime mapping is not one-to-one")
        predicted = model.predict(prediction_features)
        fit_end = training.index[-1]
        if not bool((predicted.index > fit_end).all()):
            raise ValueError("causal boundary violated: fit_end must precede every prediction")

        refit_id = len(refits)
        predicted["refit_id"] = refit_id
        predicted["fit_end"] = fit_end
        predicted_blocks.append(predicted)
        refits.append(
            RefitEvidence(
                refit_id=refit_id,
                training_start=str(training.index[0]),
                training_end=str(fit_end),
                prediction_start=str(predicted.index[0]),
                prediction_end=str(predicted.index[-1]),
                training_rows=len(training),
                prediction_rows=len(predicted),
                regime_names=regime_names,
            )
        )
        cursor = prediction_end

    predictions = pd.concat(predicted_blocks)
    return WalkForwardResult(
        predictions=predictions,
        refits=tuple(refits),
        policy=resolved_policy,
        feature_sha256=_feature_digest(frame),
    )


def analyze_prices_walk_forward(
    prices: pd.Series,
    *,
    feature_window: int = 20,
    policy: WalkForwardPolicy | None = None,
) -> WalkForwardResult:
    if (
        isinstance(feature_window, bool)
        or not isinstance(feature_window, int)
        or feature_window < 2
    ):
        raise ValueError("feature_window must be an integer of at least 2")
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("price timestamps must be timezone-aware")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price timestamps must be strictly increasing and unique")
    if len(prices) > MAX_FEATURE_ROWS:
        raise ValueError(f"price row count cannot exceed {MAX_FEATURE_ROWS}")
    try:
        numeric_prices = prices.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be numeric") from exc
    if not np.isfinite(numeric_prices.to_numpy()).all() or bool((numeric_prices <= 0).any()):
        raise ValueError("prices must be finite and positive")
    features = regime_features(numeric_prices.copy(), window=feature_window)
    return classify_walk_forward(features, policy)


def evidence_json(result: WalkForwardResult) -> str:
    return json.dumps(result.evidence(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
