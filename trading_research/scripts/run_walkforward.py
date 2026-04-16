"""
run_walkforward.py
------------------
Run walk-forward validation for a strategy and print fold-by-fold results.

Walk-forward validation slides a training window forward and tests on
the next out-of-sample period. This gives a more realistic estimate of
out-of-sample performance than a single train/val split.

Important: walk-forward runs ONLY on the train+validation portion of data.
The test set is never touched here.

Usage
~~~~~
    python scripts/run_walkforward.py
    python scripts/run_walkforward.py --config configs/base.yaml --strategy momentum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_research.config import load_config, default_config
from trading_research.data_loader import load_from_config, pivot_ohlcv
from trading_research.features import build_feature_matrix
from trading_research.strategy import build_strategy
from trading_research.portfolio import compute_weights_vectorised
from trading_research.backtest import run_backtest
from trading_research.scoring import score_backtest
from trading_research.validation import time_split, walk_forward_folds
from trading_research.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation.")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["momentum", "mean_reversion", "ma_crossover"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    import logging
    logger = logging.getLogger(__name__)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    if args.strategy:
        cfg.strategy.name = args.strategy

    print(f"\n{'='*60}")
    print(" WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    print(f" Strategy     : {cfg.strategy.name}")
    print(f" Train window : {cfg.splits.walkforward_train_years} years")
    print(f" Val window   : {cfg.splits.walkforward_val_years} years")
    print(f" Step size    : {cfg.splits.walkforward_step_months} months")
    print(f"{'='*60}\n")

    # Load data and restrict to train+val (NO test leakage)
    df = load_from_config(cfg)
    data_split = time_split(df, cfg.splits)
    train_val_df = pd.concat(
        [data_split.train, data_split.validation], ignore_index=True
    ).sort_values("date")

    # Generate walk-forward folds
    folds = walk_forward_folds(train_val_df, cfg.splits)

    if not folds:
        print("No walk-forward folds could be generated (not enough data).")
        return

    print(f"Generated {len(folds)} folds.\n")

    results = []
    strategy = build_strategy(cfg.strategy)

    for fold in folds:
        val_df = fold.validation

        close = pivot_ohlcv(val_df, "close")
        volume = pivot_ohlcv(val_df, "volume")
        open_prices = pivot_ohlcv(val_df, "open")

        features = build_feature_matrix(close, volume, cfg.features)
        signals = strategy.generate_signals(close, features)
        weights = compute_weights_vectorised(signals, cfg.portfolio)
        bt_result = run_backtest(weights, close, open_prices, cfg.backtest)
        score = score_backtest(bt_result, cfg.scoring)

        results.append(
            {
                "fold": fold.fold_id,
                "val_start": fold.val_start.date(),
                "val_end": fold.val_end.date(),
                "passed": score.passed,
                "score": score.scalar_score,
                "ann_return": score.annualised_return,
                "sortino": score.sortino,
                "max_dd": score.max_drawdown,
                "n_trades": score.n_trades,
            }
        )

    # Print table
    header = (
        f"{'Fold':>4} {'Val Start':<12} {'Val End':<12} {'Pass':<5} "
        f"{'Score':>8} {'AnnRet':>8} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['fold']:>4} {str(r['val_start']):<12} {str(r['val_end']):<12} "
            f"{'Y' if r['passed'] else 'N':<5} "
            f"{(r['score'] or 0.0):>8.4f} "
            f"{r['ann_return']:>8.2%} "
            f"{r['sortino']:>8.3f} "
            f"{r['max_dd']:>8.2%} "
            f"{r['n_trades']:>7d}"
        )

    passed_folds = [r for r in results if r["passed"]]
    print(f"\n{len(passed_folds)}/{len(results)} folds passed.")

    if passed_folds:
        avg_score = sum(r["score"] for r in passed_folds) / len(passed_folds)
        avg_return = sum(r["ann_return"] for r in passed_folds) / len(passed_folds)
        avg_sortino = sum(r["sortino"] for r in passed_folds) / len(passed_folds)
        print(f"\nMean (passing folds):")
        print(f"  Avg score      : {avg_score:.4f}")
        print(f"  Avg ann. return: {avg_return:.2%}")
        print(f"  Avg Sortino    : {avg_sortino:.3f}")


import pandas as pd  # noqa: E402 — needed inside main

if __name__ == "__main__":
    main()
