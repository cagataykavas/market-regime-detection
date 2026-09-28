from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from regime.engine import regime_features
from regime.synthetic import synthetic_regime_market
from regime.walk_forward import (
    MAX_FEATURE_ROWS,
    WalkForwardPolicy,
    analyze_prices_walk_forward,
    classify_walk_forward,
    evidence_json,
)
from regime.walk_forward_cli import main


def _features(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    volatility = np.concatenate(
        (
            np.full(rows // 2, 0.01),
            np.full(rows - rows // 2, 0.05),
        )
    )
    return pd.DataFrame(
        {
            "return_mean": rng.normal(0.0, 0.003, rows),
            "volatility": volatility + rng.normal(0.0, 0.001, rows),
            "momentum": rng.normal(0.0, 0.03, rows),
            "drawdown": -np.abs(rng.normal(0.02, 0.01, rows)),
            "downside_vol": volatility * 0.8 + rng.normal(0.0, 0.001, rows),
        },
        index=index,
    )


def _policy(**changes) -> WalkForwardPolicy:
    values = {
        "n_regimes": 2,
        "seed": 7,
        "min_train_size": 60,
        "refit_interval": 20,
    }
    values.update(changes)
    return WalkForwardPolicy(**values)


def test_each_prediction_is_strictly_after_its_fit_boundary() -> None:
    result = classify_walk_forward(_features(), _policy())

    assert len(result.predictions) == 160
    fit_end = pd.to_datetime(result.predictions["fit_end"], utc=True)
    assert bool((result.predictions.index > fit_end).all())
    assert [refit.refit_id for refit in result.refits] == list(range(8))
    assert all(refit.training_end < refit.prediction_start for refit in result.refits)


def test_future_mutation_cannot_change_earlier_predictions() -> None:
    features = _features()
    cutoff = features.index[160]
    baseline = classify_walk_forward(features, _policy())
    mutated = features.copy()
    mutated.loc[mutated.index >= cutoff, "volatility"] *= 20.0
    mutated.loc[mutated.index >= cutoff, "return_mean"] += 2.0

    changed = classify_walk_forward(mutated, _policy())

    columns = ["component", "regime", "confidence", "refit_id", "fit_end"]
    pd.testing.assert_frame_equal(
        baseline.predictions.loc[baseline.predictions.index < cutoff, columns],
        changed.predictions.loc[changed.predictions.index < cutoff, columns],
    )
    assert baseline.feature_sha256 != changed.feature_sha256


def test_prefix_predictions_are_invariant_to_appended_history() -> None:
    features = _features()
    prefix = features.iloc[:180]

    short = classify_walk_forward(prefix, _policy())
    full = classify_walk_forward(features, _policy())

    pd.testing.assert_frame_equal(short.predictions, full.predictions.loc[short.predictions.index])


def test_rolling_training_window_is_bounded() -> None:
    result = classify_walk_forward(_features(), _policy(train_window=80, refit_interval=10))

    assert result.refits[0].training_rows == 60
    assert max(refit.training_rows for refit in result.refits) == 80
    assert result.refits[-1].training_rows == 80


def test_evidence_is_deterministic_and_does_not_contain_feature_values() -> None:
    features = _features()
    left = classify_walk_forward(features, _policy())
    right = classify_walk_forward(features.copy(), _policy())

    assert evidence_json(left) == evidence_json(right)
    evidence = left.evidence()
    assert len(evidence["feature_sha256"]) == 64
    assert len(evidence["prediction_sha256"]) == 64
    assert evidence["feature_rows"] == len(features)
    assert evidence["refit_count"] == 8
    assert "return_mean" not in evidence
    assert "volatility" not in evidence


def test_price_path_builds_causal_features_and_predictions() -> None:
    prices, _ = synthetic_regime_market(seed=23, cycles=1)
    prices.index = prices.index.tz_localize("UTC")
    features = regime_features(prices, window=10)
    policy = _policy(min_train_size=50, refit_interval=25)

    result = analyze_prices_walk_forward(prices, feature_window=10, policy=policy)

    assert len(result.predictions) == len(features) - policy.min_train_size
    assert set(result.predictions["regime"]) <= {"calm", "stress"}


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda prices: prices.set_axis(prices.index.tz_localize(None)), "timezone-aware"),
        (lambda prices: prices.iloc[::-1], "strictly increasing"),
        (lambda prices: pd.concat([prices.iloc[:1], prices]), "strictly increasing"),
        (
            lambda prices: prices.mask(prices.index == prices.index[-1], np.inf),
            "finite and positive",
        ),
        (lambda prices: prices.mask(prices.index == prices.index[-1], 0.0), "finite and positive"),
    ],
)
def test_malformed_price_evidence_fails_closed(mutator, error) -> None:
    prices, _ = synthetic_regime_market(seed=29, cycles=1)
    prices.index = prices.index.tz_localize("UTC")

    with pytest.raises((TypeError, ValueError), match=error):
        analyze_prices_walk_forward(
            mutator(prices),
            feature_window=10,
            policy=_policy(min_train_size=50),
        )


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda frame: frame.rename(columns={"momentum": "wrong"}), "exactly"),
        (lambda frame: frame.set_axis(frame.index.tz_localize(None)), "timezone-aware"),
        (lambda frame: frame.iloc[::-1], "strictly increasing"),
        (lambda frame: pd.concat([frame.iloc[:1], frame]), "strictly increasing"),
        (lambda frame: frame.assign(volatility=np.inf), "finite"),
    ],
)
def test_malformed_feature_evidence_fails_closed(mutator, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        classify_walk_forward(mutator(_features()), _policy())


def test_input_features_are_not_mutated() -> None:
    features = _features()
    expected = features.copy(deep=True)

    classify_walk_forward(features, _policy())

    pd.testing.assert_frame_equal(features, expected)


def test_refit_budget_is_enforced() -> None:
    with pytest.raises(ValueError, match="refit budget"):
        classify_walk_forward(_features(), _policy(refit_interval=10, max_refits=2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_regimes": 4},
        {"seed": -1},
        {"min_train_size": 10},
        {"refit_interval": 0},
        {"train_window": 59},
        {"max_feature_rows": MAX_FEATURE_ROWS + 1},
        {"max_refits": 0},
    ],
)
def test_invalid_policies_fail_closed(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        _policy(**kwargs)


def test_cli_writes_atomic_bounded_evidence(tmp_path, capsys) -> None:
    prices, _ = synthetic_regime_market(seed=31, cycles=1)
    input_path = tmp_path / "prices.csv"
    output_path = tmp_path / "evidence.json"
    pd.DataFrame({"timestamp": prices.index, "close": prices.to_numpy()}).to_csv(
        input_path,
        index=False,
    )

    code = main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--feature-window",
            "10",
            "--min-train-size",
            "50",
            "--refit-interval",
            "25",
            "--regimes",
            "2",
        ]
    )

    assert code == 0
    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == "market-regime-walk-forward/v1"
    assert evidence["prediction_rows"] > 0
    assert json.loads(capsys.readouterr().out)["output"] == str(output_path)


def test_cli_rejects_unsorted_or_duplicate_market_data(tmp_path, capsys) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
            "close": [100.0, 101.0],
        }
    ).to_csv(path, index=False)

    assert main([str(path)]) == 3
    assert "strictly increasing" in json.loads(capsys.readouterr().out)["message"]


def test_large_but_bounded_walk_forward_run() -> None:
    result = classify_walk_forward(
        _features(2_000),
        _policy(min_train_size=200, refit_interval=100, train_window=500),
    )

    assert len(result.predictions) == 1_800
    assert len(result.refits) == 18
    assert max(refit.training_rows for refit in result.refits) == 500
