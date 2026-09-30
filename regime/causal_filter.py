from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import pairwise

import numpy as np
import pandas as pd

from regime.engine import FEATURE_COLUMNS, RegimeModel

REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REVISIONS = {"head", "latest", "main", "master", "stable"}


class CausalFilterError(ValueError):
    """Malformed or operationally untrustworthy regime-filter evidence."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FilterPolicy:
    min_training_rows: int = 60
    max_training_rows: int = 250_000
    max_prediction_rows: int = 250_000
    max_components: int = 16
    max_operations: int = 20_000_000
    min_probability: float = 1e-15

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_training_rows", self.min_training_rows, 2, 100_000),
            ("max_training_rows", self.max_training_rows, 2, 1_000_000),
            ("max_prediction_rows", self.max_prediction_rows, 1, 1_000_000),
            ("max_components", self.max_components, 2, 256),
            ("max_operations", self.max_operations, 1, 1_000_000_000),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if self.min_training_rows > self.max_training_rows:
            raise ValueError("min_training_rows cannot exceed max_training_rows")
        if (
            not isinstance(self.min_probability, (int, float))
            or not math.isfinite(self.min_probability)
            or not 0 < self.min_probability < 1
        ):
            raise ValueError("min_probability must be finite and in (0, 1)")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CausalFilterContract:
    schema_version: int
    model_revision: str
    component_names: tuple[str, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    initial_probabilities: tuple[float, ...]
    mixture_weights: tuple[float, ...]
    training_end: datetime
    training_rows: int
    pseudocount: float
    training_sha256: str
    policy_sha256: str
    contract_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "component_names": list(self.component_names),
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "initial_probabilities": list(self.initial_probabilities),
            "mixture_weights": list(self.mixture_weights),
            "training_end": _timestamp(self.training_end),
            "training_rows": self.training_rows,
            "pseudocount": self.pseudocount,
            "training_sha256": self.training_sha256,
            "policy_sha256": self.policy_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "contract_sha256": self.contract_sha256}

    def with_digest(self) -> CausalFilterContract:
        return replace(self, contract_sha256=_digest(self.body()))


@dataclass(frozen=True)
class FilteredRegimePath:
    frame: pd.DataFrame
    contract_sha256: str
    prediction_features_sha256: str
    raw_posteriors_sha256: str
    filtered_posteriors_sha256: str
    rows: int
    first_prediction: datetime
    last_prediction: datetime
    evidence_sha256: str

    def evidence(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract_sha256": self.contract_sha256,
            "prediction_features_sha256": self.prediction_features_sha256,
            "raw_posteriors_sha256": self.raw_posteriors_sha256,
            "filtered_posteriors_sha256": self.filtered_posteriors_sha256,
            "rows": self.rows,
            "first_prediction": _timestamp(self.first_prediction),
            "last_prediction": _timestamp(self.last_prediction),
            "evidence_sha256": self.evidence_sha256,
        }


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CausalFilterError("NON_CANONICAL_EVIDENCE") from exc
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CausalFilterError("TIMEZONE_AWARE_TIMESTAMP_REQUIRED")
    return value.isoformat()


def _validate_revision(value: str) -> None:
    if (
        not isinstance(value, str)
        or not REVISION.fullmatch(value)
        or value.casefold() in MUTABLE_REVISIONS
    ):
        raise CausalFilterError("INVALID_MODEL_REVISION")


def _validate_features(
    features: pd.DataFrame,
    *,
    minimum: int,
    maximum: int,
    code_prefix: str,
) -> None:
    if not isinstance(features, pd.DataFrame):
        raise CausalFilterError(f"{code_prefix}_FEATURES_REQUIRED")
    if not minimum <= len(features) <= maximum:
        raise CausalFilterError(f"{code_prefix}_ROW_BUDGET_VIOLATION")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise CausalFilterError(f"{code_prefix}_DATETIME_INDEX_REQUIRED")
    if features.index.tz is None:
        raise CausalFilterError(f"{code_prefix}_TIMEZONE_REQUIRED")
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise CausalFilterError(f"{code_prefix}_TIME_ORDER_INVALID")
    if any(column not in features for column in FEATURE_COLUMNS):
        raise CausalFilterError(f"{code_prefix}_FEATURE_COLUMN_MISSING")
    values = features[FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise CausalFilterError(f"{code_prefix}_NON_FINITE_FEATURE")


def _feature_digest(features: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for timestamp, row in features[FEATURE_COLUMNS].iterrows():
        encoded_timestamp = _timestamp(timestamp.to_pydatetime()).encode("utf-8")
        digest.update(struct.pack(">I", len(encoded_timestamp)))
        digest.update(encoded_timestamp)
        for value in row:
            digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype=">f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_probability_vector(values: np.ndarray, size: int, code: str) -> None:
    if values.shape != (size,) or not np.isfinite(values).all() or np.any(values < 0):
        raise CausalFilterError(code)
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-9, rel_tol=1e-9):
        raise CausalFilterError(code)


def _validate_contract(contract: CausalFilterContract, policy: FilterPolicy) -> None:
    if contract.schema_version != 1 or contract.contract_sha256 != _digest(
        contract.body()
    ):
        raise CausalFilterError("CONTRACT_INTEGRITY_MISMATCH")
    _validate_revision(contract.model_revision)
    components = len(contract.component_names)
    if not 2 <= components <= policy.max_components:
        raise CausalFilterError("COMPONENT_BUDGET_VIOLATION")
    if len(set(contract.component_names)) != components or any(
        not isinstance(name, str) or not name or len(name) > 128
        for name in contract.component_names
    ):
        raise CausalFilterError("INVALID_COMPONENT_NAMES")
    if (
        not policy.min_training_rows
        <= contract.training_rows
        <= policy.max_training_rows
    ):
        raise CausalFilterError("CONTRACT_TRAINING_ROW_BUDGET_VIOLATION")
    if (
        not isinstance(contract.pseudocount, (int, float))
        or not math.isfinite(contract.pseudocount)
        or contract.pseudocount <= 0
    ):
        raise CausalFilterError("INVALID_CONTRACT_PSEUDOCOUNT")
    if not SHA256.fullmatch(contract.training_sha256):
        raise CausalFilterError("INVALID_TRAINING_DIGEST")
    if contract.policy_sha256 != _digest(policy.as_dict()):
        raise CausalFilterError("POLICY_MISMATCH")
    matrix = np.asarray(contract.transition_matrix, dtype=float)
    initial = np.asarray(contract.initial_probabilities, dtype=float)
    weights = np.asarray(contract.mixture_weights, dtype=float)
    if (
        matrix.shape != (components, components)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
    ):
        raise CausalFilterError("INVALID_TRANSITION_MATRIX")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9, rtol=1e-9):
        raise CausalFilterError("INVALID_TRANSITION_MATRIX")
    _validate_probability_vector(initial, components, "INVALID_INITIAL_PROBABILITIES")
    _validate_probability_vector(weights, components, "INVALID_MIXTURE_WEIGHTS")
    if np.any(weights < policy.min_probability):
        raise CausalFilterError("MIXTURE_WEIGHT_TOO_SMALL")
    _timestamp(contract.training_end)


def fit_causal_filter(
    model: RegimeModel,
    training_features: pd.DataFrame,
    *,
    model_revision: str,
    pseudocount: float = 1.0,
    policy: FilterPolicy | None = None,
) -> CausalFilterContract:
    """Fit a transition prior from a strictly historical feature window."""

    active_policy = policy or FilterPolicy()
    _validate_revision(model_revision)
    _validate_features(
        training_features,
        minimum=active_policy.min_training_rows,
        maximum=active_policy.max_training_rows,
        code_prefix="TRAINING",
    )
    if (
        not isinstance(pseudocount, (int, float))
        or not math.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise CausalFilterError("INVALID_PSEUDOCOUNT")
    components = int(model.model.n_components)
    if not 2 <= components <= active_policy.max_components:
        raise CausalFilterError("COMPONENT_BUDGET_VIOLATION")
    if set(model.component_names) != set(range(components)):
        raise CausalFilterError("INCOMPLETE_COMPONENT_NAMES")
    operation_count = len(training_features) * components
    if operation_count > active_policy.max_operations:
        raise CausalFilterError("OPERATION_BUDGET_EXCEEDED")

    scaled = model.scaler.transform(training_features[FEATURE_COLUMNS])
    labels = np.asarray(model.model.predict(scaled), dtype=int)
    if (
        labels.shape != (len(training_features),)
        or np.any(labels < 0)
        or np.any(labels >= components)
    ):
        raise CausalFilterError("INVALID_TRAINING_COMPONENT")
    counts = np.full((components, components), float(pseudocount), dtype=float)
    for current, nxt in pairwise(labels):
        counts[current, nxt] += 1.0
    matrix = counts / counts.sum(axis=1, keepdims=True)
    initial_counts = np.bincount(labels, minlength=components).astype(float) + float(
        pseudocount
    )
    initial = initial_counts / initial_counts.sum()
    weights = np.asarray(model.model.weights_, dtype=float)
    _validate_probability_vector(weights, components, "INVALID_MIXTURE_WEIGHTS")
    if np.any(weights < active_policy.min_probability):
        raise CausalFilterError("MIXTURE_WEIGHT_TOO_SMALL")

    contract = CausalFilterContract(
        schema_version=1,
        model_revision=model_revision,
        component_names=tuple(
            model.component_names[index] for index in range(components)
        ),
        transition_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        initial_probabilities=tuple(float(value) for value in initial),
        mixture_weights=tuple(float(value) for value in weights),
        training_end=training_features.index[-1].to_pydatetime(),
        training_rows=len(training_features),
        pseudocount=float(pseudocount),
        training_sha256=_feature_digest(training_features),
        policy_sha256=_digest(active_policy.as_dict()),
        contract_sha256="",
    )
    return contract.with_digest()


def causal_forward_filter(
    raw_posteriors: np.ndarray,
    contract: CausalFilterContract,
    *,
    policy: FilterPolicy | None = None,
) -> np.ndarray:
    """Apply forward-only Bayes filtering; never use observations after the current row."""

    active_policy = policy or FilterPolicy()
    _validate_contract(contract, active_policy)
    values = np.asarray(raw_posteriors, dtype=float)
    components = len(contract.component_names)
    if values.ndim != 2 or values.shape[1] != components or len(values) < 1:
        raise CausalFilterError("INVALID_POSTERIOR_SHAPE")
    if len(values) > active_policy.max_prediction_rows:
        raise CausalFilterError("PREDICTION_ROW_BUDGET_VIOLATION")
    if len(values) * components * components > active_policy.max_operations:
        raise CausalFilterError("OPERATION_BUDGET_EXCEEDED")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise CausalFilterError("INVALID_RAW_POSTERIOR")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9, rtol=1e-9):
        raise CausalFilterError("INVALID_RAW_POSTERIOR")

    transition = np.asarray(contract.transition_matrix, dtype=float)
    weights = np.asarray(contract.mixture_weights, dtype=float)
    prior = np.asarray(contract.initial_probabilities, dtype=float)
    filtered = np.empty_like(values)
    for index, posterior in enumerate(values):
        # GMM posterior already includes its training mixture prior. Dividing by the
        # fitted weights recovers component likelihoods up to a shared normalizer.
        likelihood = posterior / weights
        unnormalized = prior * likelihood
        denominator = float(unnormalized.sum())
        if (
            not math.isfinite(denominator)
            or denominator < active_policy.min_probability
        ):
            raise CausalFilterError("DEGENERATE_FILTER_NORMALIZER")
        current = unnormalized / denominator
        filtered[index] = current
        prior = current @ transition
    return filtered


def predict_causal_regimes(
    model: RegimeModel,
    contract: CausalFilterContract,
    features: pd.DataFrame,
    *,
    model_revision: str,
    policy: FilterPolicy | None = None,
) -> FilteredRegimePath:
    """Predict GMM posteriors and apply a historical transition prior without smoothing."""

    active_policy = policy or FilterPolicy()
    _validate_contract(contract, active_policy)
    _validate_revision(model_revision)
    if model_revision != contract.model_revision:
        raise CausalFilterError("MODEL_REVISION_MISMATCH")
    components = len(contract.component_names)
    model_names = tuple(model.component_names.get(index) for index in range(components))
    model_weights = np.asarray(model.model.weights_, dtype=float)
    if (
        int(model.model.n_components) != components
        or model_names != contract.component_names
        or not np.array_equal(model_weights, np.asarray(contract.mixture_weights))
    ):
        raise CausalFilterError("MODEL_CONTRACT_MISMATCH")
    _validate_features(
        features,
        minimum=1,
        maximum=active_policy.max_prediction_rows,
        code_prefix="PREDICTION",
    )
    if features.index[0].to_pydatetime() <= contract.training_end:
        raise CausalFilterError("TRAINING_PREDICTION_OVERLAP")

    scaled = model.scaler.transform(features[FEATURE_COLUMNS])
    raw = np.asarray(model.model.predict_proba(scaled), dtype=float)
    filtered = causal_forward_filter(raw, contract, policy=active_policy)
    raw_component = raw.argmax(axis=1)
    component = filtered.argmax(axis=1)
    result = features.copy()
    result["raw_component"] = raw_component
    result["component"] = component
    result["raw_confidence"] = raw.max(axis=1)
    result["confidence"] = filtered.max(axis=1)
    result["regime"] = [contract.component_names[index] for index in component]
    for index, name in enumerate(contract.component_names):
        result[f"probability_{name}"] = filtered[:, index]

    raw_digest = _array_digest(raw)
    filtered_digest = _array_digest(filtered)
    feature_digest = _feature_digest(features)
    first_prediction = features.index[0].to_pydatetime()
    last_prediction = features.index[-1].to_pydatetime()
    evidence_body = {
        "schema_version": 1,
        "contract_sha256": contract.contract_sha256,
        "raw_posteriors_sha256": raw_digest,
        "filtered_posteriors_sha256": filtered_digest,
        "prediction_features_sha256": feature_digest,
        "rows": len(features),
        "first_prediction": _timestamp(first_prediction),
        "last_prediction": _timestamp(last_prediction),
    }
    return FilteredRegimePath(
        frame=result,
        contract_sha256=contract.contract_sha256,
        prediction_features_sha256=feature_digest,
        raw_posteriors_sha256=raw_digest,
        filtered_posteriors_sha256=filtered_digest,
        rows=len(features),
        first_prediction=first_prediction,
        last_prediction=last_prediction,
        evidence_sha256=_digest(evidence_body),
    )
