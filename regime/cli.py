from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .engine import RegimeExperiment
from .report import render_report
from .synthetic import synthetic_regime_market


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unsupervised market-regime detection reference project")
    parser.add_argument("--input", type=Path, help="optional CSV with timestamp and close columns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regimes", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--require-stable",
        action="store_true",
        help="exit 2 when the default temporal-stability policy is violated",
    )
    args = parser.parse_args(argv)

    known = None
    if args.input:
        frame = pd.read_csv(args.input, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        prices = frame["close"]
    else:
        prices, known = synthetic_regime_market(seed=args.seed, cycles=args.cycles)

    result = RegimeExperiment(n_regimes=args.regimes, seed=args.seed).analyze(prices, known)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "regimes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_report(result, args.output / "regimes.html")
    print(json.dumps({
        "rows": result["rows"],
        "regimes": list(result["regime_stats"]),
        "evaluation": result["evaluation"],
        "stability": result["stability"]["summary"],
        "stability_passed": result["stability"]["passed"],
        "output": str(args.output),
    }, indent=2))
    return 0 if not args.require_stable or result["stability"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
