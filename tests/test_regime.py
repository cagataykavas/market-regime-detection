from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.api import app
from regime.engine import RegimeExperiment
from regime.synthetic import synthetic_regime_market


def test_regime_experiment_recovers_nontrivial_structure():
    prices, labels = synthetic_regime_market(seed=42, cycles=2)
    result = RegimeExperiment(seed=42).analyze(prices, labels)
    assert len(result["regime_stats"]) == 3
    assert result["evaluation"] is not None
    assert result["evaluation"]["adjusted_rand_index"] > 0.05
    assert set(result["average_regime_duration_days"]) == set(result["regime_stats"])


def test_transition_rows_are_probabilities():
    prices, _ = synthetic_regime_market(seed=8, cycles=2)
    result = RegimeExperiment(seed=8).analyze(prices)
    for row in result["transition_matrix"].values():
        assert np.isclose(sum(row.values()), 1.0, atol=1e-6)


def test_demo_api():
    response = TestClient(app).get("/demo?seed=42")
    assert response.status_code == 200
    payload = response.json()
    assert payload["feature_rows"] > 200
    assert len(payload["timeline"]) == payload["feature_rows"]
