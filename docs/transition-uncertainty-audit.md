# Regime transition uncertainty audit

An empirical transition matrix can look precise even when some origin regimes have only a handful
of outgoing observations. `regime.transition_uncertainty` adds a fail-closed admission boundary
before transition estimates are used in downstream scenario or risk logic.

The audit:

- requires bounded, contiguous, ordered regime observations so missing periods are not treated as
  direct transitions;
- counts every origin/destination pair, including zero-count transitions;
- computes Wilson confidence intervals with Bonferroni family-wise adjustment across the complete
  transition matrix;
- requires minimum outgoing support for every regime and a maximum interval width for every cell;
- emits structured policy violations and a SHA-256 digest over the exact sequence and policy;
- returns exit code `0` for pass, `3` for policy failure, and `2` for invalid evidence.

Input is a JSON array with exact `position` and `regime` fields:

```json
[
  {"position": 0, "regime": "calm"},
  {"position": 1, "regime": "stress"}
]
```

```bash
python -m regime.transition_uncertainty observations.json \
  --confidence 0.95 \
  --min-outgoing-transitions 30 \
  --max-interval-width 0.50 \
  --output transition-uncertainty.json
```

## Limitations

Wilson intervals are an approximate diagnostic and treat outgoing transitions as binomial trials.
Market-regime sequences are dependent, estimated labels can be wrong, and transition behavior can
change over time. The audit does not establish Markov stationarity, economic meaning, causal
structure, or future transition probabilities. Thresholds require calibration on reviewed market
histories; block-bootstrap or state-space uncertainty is a stronger next step when enough data is
available.
