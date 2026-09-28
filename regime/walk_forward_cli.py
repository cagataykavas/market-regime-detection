from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .walk_forward import WalkForwardPolicy, analyze_prices_walk_forward


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe walk-forward market-regime classification",
    )
    parser.add_argument("input", type=Path, help="CSV containing timestamp and close columns")
    parser.add_argument("--output", type=Path, default=Path("artifacts/walk-forward.json"))
    parser.add_argument("--feature-window", type=int, default=20)
    parser.add_argument("--min-train-size", type=int, default=120)
    parser.add_argument("--refit-interval", type=int, default=20)
    parser.add_argument("--train-window", type=int)
    parser.add_argument("--regimes", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = pd.read_csv(args.input)
        if list(frame.columns) != ["timestamp", "close"]:
            raise ValueError("input CSV must contain exactly timestamp,close columns")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
            raise ValueError("input timestamps must be strictly increasing and unique")
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=float)
        prices = pd.Series(close, index=timestamps, name="close")
        if not np.isfinite(close).all() or (prices <= 0).any():
            raise ValueError("close prices must be finite and positive")
        policy = WalkForwardPolicy(
            n_regimes=args.regimes,
            seed=args.seed,
            min_train_size=args.min_train_size,
            refit_interval=args.refit_interval,
            train_window=args.train_window,
        )
        result = analyze_prices_walk_forward(
            prices,
            feature_window=args.feature_window,
            policy=policy,
        )
        evidence = result.evidence()
        _atomic_write_json(args.output, evidence)
        print(json.dumps({"output": str(args.output), **evidence}, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
