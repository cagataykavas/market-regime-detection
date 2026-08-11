from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def regime_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    returns = prices.pct_change()
    out = pd.DataFrame(index=prices.index)
    out["return_mean"] = returns.rolling(window).mean()
    out["volatility"] = returns.rolling(window).std()
    out["momentum"] = prices.pct_change(window)
    out["drawdown"] = prices / prices.rolling(window * 3, min_periods=window).max() - 1
    out["downside_vol"] = returns.where(returns < 0).rolling(window).std()
    return out.dropna()


def fit_regimes(features: pd.DataFrame, n_regimes: int = 3, seed: int = 42):
    scaler = StandardScaler()
    z = scaler.fit_transform(features)
    model = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=seed)
    labels = model.fit_predict(z)
    result = features.copy()
    result["regime"] = labels
    stats = result.groupby("regime").agg(
        observations=("volatility", "size"),
        mean_return=("return_mean", "mean"),
        volatility=("volatility", "mean"),
        momentum=("momentum", "mean"),
        drawdown=("drawdown", "mean"),
    )
    ordering = stats.sort_values(["volatility", "mean_return"]).index.tolist()
    names = {ordering[0]: "calm", ordering[-1]: "stress"}
    for r in ordering[1:-1]:
        names[r] = "transition"
    result["regime_name"] = result["regime"].map(names)
    return result, stats, model, scaler


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    segments = [
        rng.normal(0.0005, 0.007, 250),
        rng.normal(-0.0010, 0.025, 120),
        rng.normal(0.0002, 0.012, 220),
    ]
    r = np.concatenate(segments)
    prices = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2022-01-01", periods=len(r), freq="B"))
    regimes, stats, _, _ = fit_regimes(regime_features(prices))
    print(stats.round(4))
    print(regimes[["regime_name", "volatility", "drawdown"]].tail(10))
