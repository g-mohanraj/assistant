"""
run_baseline.py
---------------
Run all three baseline strategies and print a summary comparison.

Usage
~~~~~
    python scripts/run_baseline.py
    python scripts/run_baseline.py --config configs/base.yaml
    python scripts/run_baseline.py --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_research.config import load_config, default_config
from trading_research.experiment_runner import run_multiple_strategies
from trading_research.utils import configure_logging


BASELINE_STRATEGIES = ["momentum", "mean_reversion", "ma_crossover"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all baseline strategies.")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Data split to evaluate on",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    print(f"\n{'='*60}")
    print(" BASELINE STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f" Split       : {args.split}")
    print(f" Symbols     : {cfg.data.symbols}")
    print(f" Date range  : {cfg.data.start_date} → {cfg.data.end_date}")
    print(f"{'='*60}\n")

    records = run_multiple_strategies(
        cfg,
        strategies=BASELINE_STRATEGIES,
        split=args.split,
        notes="baseline comparison run",
    )

    # Print results table
    header = (
        f"{'Strategy':<18} {'Passed':<8} {'Score':>8} "
        f"{'Ann.Ret':>8} {'Sortino':>8} {'MaxDD':>8} "
        f"{'Turnover':>10} {'Trades':>7}"
    )
    print(header)
    print("-" * len(header))

    for rec in records:
        m = rec.get("metrics", {})
        print(
            f"{rec['strategy_name']:<18} "
            f"{'YES' if rec['passed'] else 'NO':<8} "
            f"{(rec['scalar_score'] or 0.0):>8.4f} "
            f"{m.get('annualised_return', 0.0):>8.2%} "
            f"{m.get('sortino', 0.0):>8.3f} "
            f"{m.get('max_drawdown', 0.0):>8.2%} "
            f"{m.get('annualised_turnover', 0.0):>10.2f}x "
            f"{m.get('n_trades', 0):>7d}"
        )
        if rec.get("fail_reasons"):
            for reason in rec["fail_reasons"]:
                print(f"    ✗ {reason}")

    passed = [r for r in records if r.get("passed")]
    print(f"\n{len(passed)}/{len(records)} strategies passed hard-fail checks.")

    if passed:
        best = max(passed, key=lambda r: r.get("scalar_score") or float("-inf"))
        print(f"Best strategy: {best['strategy_name']} (score={best['scalar_score']:.4f})")

    print(f"\nExperiment log: {cfg.experiment.log_path}")


if __name__ == "__main__":
    main()
