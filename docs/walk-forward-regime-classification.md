# Point-in-time walk-forward regime classification

The default demo fits one GMM to a complete history so it can visualize the entire sample. That is useful for
exploration, but it must not be presented as historical out-of-sample evidence: the scaler, mixture parameters,
and semantic component ordering have all seen future observations.

`regime.walk_forward` provides a separate point-in-time path. It begins after a declared minimum training
window, fits only rows strictly earlier than the first prediction, classifies one bounded block, and then
refits. Every prediction carries the `refit_id` and `fit_end` that produced it. Price and feature inputs must
use strictly increasing, unique, timezone-aware timestamps; the API does not silently sort point-in-time
evidence.

```python
from regime.walk_forward import WalkForwardPolicy, analyze_prices_walk_forward

result = analyze_prices_walk_forward(
    prices,
    feature_window=20,
    policy=WalkForwardPolicy(
        n_regimes=3,
        min_train_size=252,
        refit_interval=20,
        train_window=756,
    ),
)

predictions = result.predictions
evidence = result.evidence()
```

The equivalent CSV interface is:

```bash
python -m regime.walk_forward_cli prices.csv \
  --feature-window 20 \
  --min-train-size 252 \
  --refit-interval 20 \
  --train-window 756 \
  --output artifacts/walk-forward.json
```

## Evidence contract

The JSON evidence contains policy parameters, point-in-time train/predict boundaries for every refit, regime
counts, confidence summaries, and canonical SHA-256 identities for the feature and prediction sequences. It
does not persist prices or raw feature values. The full prediction frame remains available in memory for
downstream backtests.

The implementation fails closed on unordered, duplicate, timezone-naive, non-finite, undersized, or
over-budget feature evidence. It supports two or three regimes so semantic names remain one-to-one across
refits. Feature rows and refit counts have hard ceilings of 100,000 and 256 that callers cannot raise.

## Validation invariants

- `fit_end < prediction_timestamp` for every output row;
- changing future features cannot change earlier predictions;
- appending history cannot rewrite the predictions for an existing prefix;
- a rolling training window never exceeds its declared bound;
- repeated runs with the same evidence and seed produce identical digests.

## Methodological boundary

Point-in-time fitting removes this implementation's look-ahead path; it does not prove regimes are useful or
tradable. Rolling features are contemporaneous classifications, not forecasts of the next regime. GMMs still
ignore state persistence, confidence is not calibrated, and volatility-based semantic naming can become
unstable when components are poorly separated. Any trading evaluation must additionally model execution,
costs, feature availability, universe selection, and multiple testing.
