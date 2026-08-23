from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["return_mean", "volatility", "momentum", "drawdown", "downside_vol"]


def regime_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    prices = prices.astype(float).sort_index()
    returns = prices.pct_change()
    out = pd.DataFrame(index=prices.index)
    out["return_mean"] = returns.rolling(window).mean()
    out["volatility"] = returns.rolling(window).std()
    out["momentum"] = prices.pct_change(window)
    out["drawdown"] = prices / prices.rolling(window * 3, min_periods=window).max() - 1
    out["downside_vol"] = returns.where(returns < 0).rolling(window).std()
    return out.dropna()


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    states = sorted(labels.dropna().unique())
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    values = labels.dropna().tolist()
    for current, nxt in zip(values[:-1], values[1:]):
        matrix.loc[current, nxt] += 1.0
    row_sum = matrix.sum(axis=1).replace(0, 1.0)
    return matrix.div(row_sum, axis=0)


@dataclass
class RegimeModel:
    scaler: StandardScaler
    model: GaussianMixture
    component_names: dict[int, str]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        z = self.scaler.transform(features[FEATURE_COLUMNS])
        component = self.model.predict(z)
        probabilities = self.model.predict_proba(z)
        result = features.copy()
        result["component"] = component
        result["regime"] = [self.component_names[int(value)] for value in component]
        result["confidence"] = probabilities.max(axis=1)
        return result


class RegimeExperiment:
    def __init__(self, n_regimes: int = 3, seed: int = 42) -> None:
        self.n_regimes = n_regimes
        self.seed = seed

    def fit(self, features: pd.DataFrame) -> RegimeModel:
        scaler = StandardScaler()
        z = scaler.fit_transform(features[FEATURE_COLUMNS])
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=8,
            reg_covar=1e-6,
            random_state=self.seed,
        )
        components = model.fit_predict(z)
        working = features.copy()
        working["component"] = components
        stats = working.groupby("component").agg(
            volatility=("volatility", "mean"),
            mean_return=("return_mean", "mean"),
            drawdown=("drawdown", "mean"),
        )
        ordered = stats.sort_values("volatility").index.tolist()
        names: dict[int, str] = {}
        if len(ordered) == 1:
            names[int(ordered[0])] = "single"
        elif len(ordered) == 2:
            names[int(ordered[0])] = "calm"
            names[int(ordered[1])] = "stress"
        else:
            names[int(ordered[0])] = "calm"
            names[int(ordered[-1])] = "stress"
            for item in ordered[1:-1]:
                names[int(item)] = "transition"
        return RegimeModel(scaler, model, names)

    def analyze(self, prices: pd.Series, known_labels: pd.Series | None = None) -> dict[str, Any]:
        features = regime_features(prices)
        model = self.fit(features)
        classified = model.predict(features)
        stats = classified.groupby("regime").agg(
            observations=("volatility", "size"),
            mean_return=("return_mean", "mean"),
            volatility=("volatility", "mean"),
            momentum=("momentum", "mean"),
            drawdown=("drawdown", "mean"),
            confidence=("confidence", "mean"),
        )
        transitions = transition_matrix(classified["regime"])
        durations: dict[str, list[int]] = {}
        current = None
        run = 0
        for value in classified["regime"]:
            if value == current:
                run += 1
            else:
                if current is not None:
                    durations.setdefault(str(current), []).append(run)
                current = value
                run = 1
        if current is not None:
            durations.setdefault(str(current), []).append(run)
        average_duration = {key: float(np.mean(value)) for key, value in durations.items()}

        evaluation = None
        if known_labels is not None:
            aligned = known_labels.reindex(classified.index).dropna()
            predicted = classified.loc[aligned.index, "component"]
            evaluation = {"adjusted_rand_index": float(adjusted_rand_score(aligned, predicted)), "observations": len(aligned)}

        return {
            "rows": len(prices),
            "feature_rows": len(features),
            "regime_stats": {name: {key: float(value) for key, value in row.items()} for name, row in stats.iterrows()},
            "transition_matrix": {str(row): {str(col): float(transitions.loc[row, col]) for col in transitions.columns} for row in transitions.index},
            "average_regime_duration_days": average_duration,
            "evaluation": evaluation,
            "timeline": [
                {
                    "timestamp": str(index),
                    "regime": str(row["regime"]),
                    "component": int(row["component"]),
                    "confidence": float(row["confidence"]),
                    "volatility": float(row["volatility"]),
                    "drawdown": float(row["drawdown"]),
                }
                for index, row in classified.iterrows()
            ],
        }
