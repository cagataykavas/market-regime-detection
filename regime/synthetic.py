from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_NAMES = {0: "calm", 1: "transition", 2: "stress"}


def synthetic_regime_market(
    seed: int = 42,
    cycles: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Generate a piecewise market with construction-time regime labels.

    Labels exist only because this is synthetic data; they make it possible to test
    whether the unsupervised clustering captures broad state changes without claiming
    that real markets provide clean regime ground truth.
    """
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    labels: list[int] = []
    previous = 0.0

    specifications = (
        (0, 160, 0.00045, 0.0065, 0.10),
        (1, 90, 0.00010, 0.0120, 0.02),
        (2, 75, -0.00085, 0.0240, -0.10),
        (1, 85, 0.00015, 0.0135, 0.00),
        (0, 150, 0.00040, 0.0070, 0.08),
    )

    for _ in range(cycles):
        for label, length, mean, sigma, phi in specifications:
            for _step in range(length):
                innovation = rng.normal(0.0, sigma)
                value = mean + phi * previous + innovation
                returns.append(value)
                labels.append(label)
                previous = value

    index = pd.date_range("2019-01-02", periods=len(returns), freq="B")
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index, name="close")
    ground_truth = pd.Series(labels, index=index, name="synthetic_regime")
    return prices, ground_truth
