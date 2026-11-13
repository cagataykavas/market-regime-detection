# Regime occupancy drift gate

This gate detects when the mix of named market regimes in a monitoring window has moved materially away from a reviewed reference window. It is intentionally separate from transition-matrix validation: transition tests ask how states move between one another, while this audit asks how much time the system spends in each state.

## Evidence contract

The command consumes counts rather than raw prices, returns or timestamps:

```json
{
  "schema_version": 1,
  "governed_states": ["calm", "transition", "stress"],
  "reference": {
    "counts": {"calm": 500, "transition": 300, "stress": 200},
    "effective_sample_size": 350
  },
  "current": {
    "counts": {"calm": 100, "transition": 60, "stress": 40},
    "effective_sample_size": 70
  }
}
```

Every governed state must be present, including explicit zero counts. `effective_sample_size` is optional and defaults to the observation count. Supply a smaller reviewed value when persistence or overlapping rolling features make daily state labels dependent.

The reference and current counts must come from the **same frozen model, scaler and semantic component mapping**. Comparing labels from independently refitted GMMs is not a valid drift test even if both use names such as `calm` and `stress`.

## Decision logic

The default policy fails when any of the following is true:

- the reference has fewer than 250 observations;
- the current window has fewer than 60 observations;
- Jeffreys-smoothed Jensen–Shannon divergence exceeds 0.05;
- any raw regime share moves by more than 15 percentage points;
- a regime absent from the reference exceeds 5% of the current window.

The report also includes a simultaneous Hoeffding sampling bound:

```text
sqrt(log(2K/alpha) / (2 n_ref_eff))
  + sqrt(log(2K/alpha) / (2 n_cur_eff))
```

where `K` is the number of governed regimes. A state is marked `statistically_resolved` when its absolute share shift exceeds this bound. This diagnostic does not replace the explicit effect-size gates: a very large sample can resolve a change too small to matter operationally.

Both evidence and decisions receive canonical SHA-256 digests. State order and JSON object order therefore do not change the audit identity.

## Usage

```bash
python -m regime.occupancy_drift examples/occupancy-evidence.json \
  --output artifacts/occupancy-drift.json
```

Exit codes:

- `0`: evidence is valid and policy passes;
- `1`: evidence is valid but drift policy rejects it;
- `2`: input, schema or policy is malformed.

Thresholds can be overridden explicitly:

```bash
python -m regime.occupancy_drift examples/occupancy-evidence.json \
  --max-js 0.04 \
  --max-share-shift 0.10 \
  --max-emergent-share 0.02
```

The loader rejects duplicate keys, non-finite JSON numbers, symbolic links, files larger than 1 MiB, booleans masquerading as counts, unknown states and invalid effective sample sizes. Reports are atomically replaced.

## Limitations

- GMM regime names are heuristic. This gate cannot determine whether a semantic label remains economically correct.
- The Hoeffding diagnostic assumes independent bounded observations after accounting for the supplied effective sample size. Selecting that value is a governance decision; it is not estimated here.
- Occupancy drift can pass while transition behavior, feature distributions or within-regime returns drift materially.
- A digest binds bytes and policy output but does not authenticate the producer. Signed evidence and immutable storage are deployment responsibilities.
- Default limits are engineering examples, not trading, capital or regulatory recommendations.

The next useful layer is a joint release report that requires both occupancy and transition-uncertainty evidence from the same frozen model and monitoring window.
