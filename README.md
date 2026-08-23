# Market Regime Detection

A runnable unsupervised market-state project using **Gaussian Mixture Models (GMMs)** over rolling return, volatility, momentum, drawdown and downside-volatility features.

The repository adds the engineering pieces around the clustering itself: deterministic synthetic ground truth, confidence scores, transition matrices, regime-duration analysis, API/CLI entrypoints, reports, Docker and CI.

## Architecture

```mermaid
flowchart LR
    P[Price series] --> F[Rolling market-state features]
    F --> S[StandardScaler]
    S --> G[Gaussian Mixture Model]
    G --> C[Component probabilities]
    C --> N[Semantic naming by volatility]
    N --> T[Transition matrix]
    N --> D[Contiguous duration analysis]
    N --> R[Regime statistics]
    R --> REP[JSON + HTML report]
    T --> REP
    D --> REP
    N --> API[FastAPI]
```

## Features

- rolling mean return;
- rolling volatility;
- momentum;
- rolling drawdown;
- downside volatility.

The model is fitted after standardization. GMM component IDs are arbitrary, so the project explicitly maps components to semantic names by their average volatility:

```text
lowest volatility  -> calm
middle components  -> transition
highest volatility -> stress
```

This avoids pretending that GMM component `0` has a stable business meaning across fits.

## Synthetic validation

`regime/synthetic.py` generates piecewise calm, transition and stress periods with construction-time labels. The unsupervised predictions are compared with those labels using **Adjusted Rand Index (ARI)**.

That evaluation is useful for CI and methodology checks. Real financial markets do not provide clean objective regime labels, so the synthetic ARI is not presented as real-world accuracy.

## Transition analysis

After classifying each timestamp, the project measures empirical transitions such as:

```text
calm -> calm
calm -> transition
transition -> stress
stress -> transition
```

It also measures average contiguous duration by named regime.

Important distinction: **this is still a GMM, not an HMM**. Temporal transition probabilities are measured after clustering; they are not part of the likelihood optimized by the mixture model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m regime.cli --seed 42 --regimes 3 --cycles 2
```

Artifacts:

```text
artifacts/regimes.json
artifacts/regimes.html
```

Custom CSV input:

```bash
python -m regime.cli --input market.csv
```

with:

```text
timestamp, close
```

## API

```bash
uvicorn app.api:app --reload
curl http://localhost:8000/demo
```

`POST /analyze` accepts a custom price history and number of regimes.

## Docker

```bash
docker build -t market-regime-detection .
docker run --rm -p 8000:8000 market-regime-detection
```

## Tests / CI

```bash
ruff check .
pytest -q
```

CI runs synthetic regime validation, probability-transition checks, a CLI experiment and container build.

## Model limitations

GMMs assume observations come from a finite mixture of parametric distributions and do not directly model state persistence. Rolling features are autocorrelated, component naming is heuristic, and real markets can transition gradually or create regimes absent from the training sample. Extensions worth comparing include HMMs, change-point detection, switching autoregressions and nonparametric clustering.

## Portfolio signal

**Python · Pandas · scikit-learn · Gaussian Mixture Models · unsupervised learning · market regimes · transition analysis · ARI · FastAPI · Docker · CI/CD**
