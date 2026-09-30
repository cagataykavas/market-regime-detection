# Forward-only Bayesian regime filtering

The base detector fits a Gaussian Mixture Model (GMM) and classifies each timestamp independently.
Its reported transition matrix describes label history after classification; it does not influence
predictions. That makes isolated low-confidence regime flips easy to produce even when the training
history shows persistent states.

`regime.causal_filter` adds a model-independent, forward-only state filter. It learns a smoothed
component transition matrix from a strictly historical training window, then combines that prior
with each new GMM observation in timestamp order:

```text
predicted_prior[t] = filtered[t-1] @ transition_matrix
component_likelihood[t] ∝ gmm_posterior[t] / gmm_mixture_weight
filtered[t] ∝ predicted_prior[t] * component_likelihood[t]
```

Dividing by the fitted GMM mixture weight matters: `predict_proba` already contains that training
prior. Multiplying the posterior directly by the transition prior would count the mixture prior
twice.

## Usage

```python
from regime.causal_filter import fit_causal_filter, predict_causal_regimes
from regime.engine import RegimeExperiment

model = RegimeExperiment(n_regimes=3, seed=42).fit(training_features)
contract = fit_causal_filter(
    model,
    training_features,
    model_revision="sha256:immutable-model-id",
)
result = predict_causal_regimes(
    model,
    contract,
    future_features,
    model_revision="sha256:immutable-model-id",
)
```

The prediction window must start strictly after the contract's training boundary. The output keeps
both raw and filtered components/confidence plus a normalized probability column for every semantic
regime. Contract and result evidence bind the model revision, historical feature bytes, transition
policy, raw posteriors and filtered posteriors without reproducing market values.

## Operational properties

- filtering is prefix-invariant: appending future observations cannot change earlier states;
- only the forward recursion is implemented—there is no backward smoothing or look-ahead;
- Dirichlet pseudocounts prevent unseen transitions from becoming impossible forever;
- timestamps must be unique, ordered and timezone-aware;
- training rows, prediction rows, components and matrix operations are bounded;
- non-finite features/posteriors, mutable model aliases and contract tampering fail closed;
- semantic regime names remain tied to the fitted GMM component mapping.

This filter complements a walk-forward estimator: each point-in-time GMM fit can build its transition
contract from the same historical window and apply it only to the subsequent prediction block.

## Limitations

The Markov assumption is an approximation, GMM component assignments are uncertain, and the
transition matrix inherits the training window's regime mix. Filtering can delay detection of a
genuine abrupt break. A stable label is not proof of trading value, causal market structure or
future performance. The model revision is caller-supplied and content-bound, not authenticated.

The next step is to integrate the filter into the causal walk-forward path, calibrate pseudocounts
and alert latency on held-out regimes, and compare log score against independent GMM posteriors and
a fully trained Hidden Markov Model.
