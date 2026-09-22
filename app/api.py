from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from regime.engine import RegimeExperiment
from regime.synthetic import synthetic_regime_market

app = FastAPI(title="Market Regime Detection", version="1.0.0")


class PricePoint(BaseModel):
    timestamp: str
    close: float = Field(gt=0)


class RegimeRequest(BaseModel):
    prices: list[PricePoint] = Field(min_length=120)
    n_regimes: int = Field(default=3, ge=2, le=6)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo(seed: int = 42) -> dict[str, object]:
    prices, labels = synthetic_regime_market(seed=seed, cycles=2)
    return RegimeExperiment(seed=seed).analyze(prices, labels)


@app.post("/analyze")
def analyze(payload: RegimeRequest) -> dict[str, object]:
    try:
        index = pd.to_datetime([item.timestamp for item in payload.prices], utc=True)
        prices = pd.Series([item.close for item in payload.prices], index=index, name="close").sort_index()
        return RegimeExperiment(n_regimes=payload.n_regimes).analyze(prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
