from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime.causal_filter import (
    CausalFilterContract,
    CausalFilterError,
    FilterPolicy,
    causal_forward_filter,
    fit_causal_filter,
    predict_causal_regimes,
)
from regime.engine import RegimeExperiment, regime_features
from regime.synthetic import synthetic_regime_market


def manual_contract(**overrides: object) -> CausalFilterContract:
    policy_digest = hashlib.sha256(
        json.dumps(
            FilterPolicy().as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    values = {
        "schema_version": 1,
        "model_revision": "sha256:gmm-20260930",
        "component_names": ("calm", "stress"),
        "transition_matrix": ((0.9, 0.1), (0.2, 0.8)),
        "initial_probabilities": (0.6, 0.4),
        "mixture_weights": (0.5, 0.5),
        "training_end": pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        "training_rows": 100,
        "pseudocount": 1.0,
        "training_sha256": "a" * 64,
        "policy_sha256": policy_digest,
        "contract_sha256": "",
        **overrides,
    }
    return CausalFilterContract(**values).with_digest()


def fitted_fixture():
    prices, _ = synthetic_regime_market(seed=42, cycles=4)
    prices.index = prices.index.tz_localize("UTC")
    features = regime_features(prices)
    split = len(features) // 2
    training = features.iloc[:split]
    prediction = features.iloc[split:]
    model = RegimeExperiment(n_regimes=3, seed=42).fit(training)
    contract = fit_causal_filter(
        model,
        training,
        model_revision="sha256:synthetic-gmm-42",
    )
    return model, contract, training, prediction


def test_forward_filter_matches_hand_calculation() -> None:
    contract = manual_contract()
    raw = np.array([[0.8, 0.2], [0.3, 0.7]])

    filtered = causal_forward_filter(raw, contract)

    expected_first = np.array([0.8571428571428571, 0.14285714285714288])
    next_prior = expected_first @ np.array(contract.transition_matrix)
    likelihood = raw[1] / np.array(contract.mixture_weights)
    expected_second = next_prior * likelihood
    expected_second /= expected_second.sum()
    assert np.allclose(filtered[0], expected_first)
    assert np.allclose(filtered[1], expected_second)


def test_filter_removes_gmm_training_prior_before_transition_prior() -> None:
    contract = manual_contract(
        initial_probabilities=(0.5, 0.5),
        mixture_weights=(0.8, 0.2),
    )
    raw = np.array([[0.8, 0.2]])

    filtered = causal_forward_filter(raw, contract)

    assert np.allclose(filtered[0], [0.5, 0.5])


def test_forward_filter_is_prefix_invariant() -> None:
    contract = manual_contract()
    prefix = np.array([[0.8, 0.2], [0.3, 0.7], [0.55, 0.45]])
    extended = np.vstack([prefix, [0.01, 0.99], [0.99, 0.01]])

    prefix_result = causal_forward_filter(prefix, contract)
    extended_result = causal_forward_filter(extended, contract)

    assert np.array_equal(prefix_result, extended_result[: len(prefix)])


def test_real_gmm_integration_produces_normalized_causal_path() -> None:
    model, contract, _, prediction = fitted_fixture()

    result = predict_causal_regimes(
        model,
        contract,
        prediction,
        model_revision="sha256:synthetic-gmm-42",
    )

    probability_columns = [
        column for column in result.frame if column.startswith("probability_")
    ]
    assert len(result.frame) == len(prediction)
    assert np.allclose(result.frame[probability_columns].sum(axis=1), 1.0)
    assert set(result.frame["regime"]).issubset(set(contract.component_names))
    assert result.frame.index.equals(prediction.index)
    assert len(result.evidence_sha256) == 64


def test_real_prediction_path_is_prefix_invariant() -> None:
    model, contract, _, prediction = fitted_fixture()
    prefix = prediction.iloc[:80]
    extended = prediction.iloc[:120]

    first = predict_causal_regimes(
        model, contract, prefix, model_revision="sha256:synthetic-gmm-42"
    )
    second = predict_causal_regimes(
        model, contract, extended, model_revision="sha256:synthetic-gmm-42"
    )

    columns = [column for column in first.frame if column.startswith("probability_")]
    assert np.array_equal(
        first.frame[columns].to_numpy(),
        second.frame.iloc[: len(prefix)][columns].to_numpy(),
    )


def test_contract_is_deterministic_and_binds_training_order() -> None:
    model, contract, training, _ = fitted_fixture()
    repeated = fit_causal_filter(
        model,
        training,
        model_revision="sha256:synthetic-gmm-42",
    )

    assert contract.as_dict() == repeated.as_dict()
    assert len(contract.contract_sha256) == 64
    shuffled = training.iloc[::-1]
    with pytest.raises(CausalFilterError, match="TRAINING_TIME_ORDER_INVALID"):
        fit_causal_filter(model, shuffled, model_revision="sha256:synthetic-gmm-42")


def test_transition_and_initial_probabilities_are_normalized() -> None:
    _, contract, _, _ = fitted_fixture()

    assert all(np.isclose(sum(row), 1.0) for row in contract.transition_matrix)
    assert np.isclose(sum(contract.initial_probabilities), 1.0)
    assert np.isclose(sum(contract.mixture_weights), 1.0)


def test_prediction_must_start_after_training() -> None:
    model, contract, training, _ = fitted_fixture()

    with pytest.raises(CausalFilterError, match="TRAINING_PREDICTION_OVERLAP"):
        predict_causal_regimes(
            model,
            contract,
            training.tail(10),
            model_revision="sha256:synthetic-gmm-42",
        )


def test_model_revision_and_contract_integrity_are_enforced() -> None:
    model, contract, _, prediction = fitted_fixture()
    with pytest.raises(CausalFilterError, match="MODEL_REVISION_MISMATCH"):
        predict_causal_regimes(
            model,
            contract,
            prediction,
            model_revision="sha256:other-model",
        )


def test_contract_policy_and_vector_dimensions_are_enforced() -> None:
    with pytest.raises(CausalFilterError, match="POLICY_MISMATCH"):
        causal_forward_filter(
            np.array([[0.5, 0.5]]),
            manual_contract(),
            policy=FilterPolicy(max_prediction_rows=10),
        )
    with pytest.raises(CausalFilterError, match="INVALID_INITIAL_PROBABILITIES"):
        causal_forward_filter(
            np.array([[0.5, 0.5]]),
            manual_contract(initial_probabilities=(1.0,)),
        )


def test_prediction_rejects_model_that_does_not_match_contract() -> None:
    model, contract, training, prediction = fitted_fixture()
    other_model = RegimeExperiment(n_regimes=3, seed=7).fit(training)

    with pytest.raises(CausalFilterError, match="MODEL_CONTRACT_MISMATCH"):
        predict_causal_regimes(
            other_model,
            contract,
            prediction,
            model_revision="sha256:synthetic-gmm-42",
        )
    tampered = replace(contract, training_rows=999)
    with pytest.raises(CausalFilterError, match="CONTRACT_INTEGRITY_MISMATCH"):
        predict_causal_regimes(
            model,
            tampered,
            prediction,
            model_revision="sha256:synthetic-gmm-42",
        )


@pytest.mark.parametrize("revision", ["latest", "main", "HEAD"])
def test_mutable_model_revisions_are_rejected(revision: str) -> None:
    model, _, training, _ = fitted_fixture()
    with pytest.raises(CausalFilterError, match="INVALID_MODEL_REVISION"):
        fit_causal_filter(model, training, model_revision=revision)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (np.array([0.5, 0.5]), "INVALID_POSTERIOR_SHAPE"),
        (np.array([[0.5, 0.4]]), "INVALID_RAW_POSTERIOR"),
        (np.array([[float("nan"), 0.5]]), "INVALID_RAW_POSTERIOR"),
        (np.array([[-0.1, 1.1]]), "INVALID_RAW_POSTERIOR"),
    ],
)
def test_malformed_posteriors_fail_closed(raw: np.ndarray, code: str) -> None:
    with pytest.raises(CausalFilterError, match=code):
        causal_forward_filter(raw, manual_contract())


def test_feature_validation_rejects_naive_duplicate_and_non_finite_evidence() -> None:
    model, _, training, _ = fitted_fixture()
    naive = training.copy()
    naive.index = naive.index.tz_localize(None)
    duplicate = pd.concat([training.iloc[:2], training.iloc[1:]])
    non_finite = training.copy()
    non_finite.iloc[0, 0] = np.inf

    with pytest.raises(CausalFilterError, match="TRAINING_TIMEZONE_REQUIRED"):
        fit_causal_filter(model, naive, model_revision="sha256:model-1")
    with pytest.raises(CausalFilterError, match="TRAINING_TIME_ORDER_INVALID"):
        fit_causal_filter(model, duplicate, model_revision="sha256:model-1")
    with pytest.raises(CausalFilterError, match="TRAINING_NON_FINITE_FEATURE"):
        fit_causal_filter(model, non_finite, model_revision="sha256:model-1")


def test_resource_budgets_fail_closed() -> None:
    model, _, training, _ = fitted_fixture()
    with pytest.raises(CausalFilterError, match="TRAINING_ROW_BUDGET_VIOLATION"):
        fit_causal_filter(
            model,
            training,
            model_revision="sha256:model-1",
            policy=FilterPolicy(max_training_rows=100),
        )
    with pytest.raises(CausalFilterError, match="OPERATION_BUDGET_EXCEEDED"):
        fit_causal_filter(
            model,
            training,
            model_revision="sha256:model-1",
            policy=FilterPolicy(max_operations=100),
        )


def test_evidence_is_recomputable_without_raw_feature_values() -> None:
    model, contract, _, prediction = fitted_fixture()
    result = predict_causal_regimes(
        model,
        contract,
        prediction.iloc[:10],
        model_revision="sha256:synthetic-gmm-42",
    )
    evidence = result.evidence()

    assert set(evidence) == {
        "schema_version",
        "contract_sha256",
        "prediction_features_sha256",
        "raw_posteriors_sha256",
        "filtered_posteriors_sha256",
        "rows",
        "first_prediction",
        "last_prediction",
        "evidence_sha256",
    }
    assert "return_mean" not in str(evidence)
