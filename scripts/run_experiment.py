"""
run_experiment.py
-----------------
Run a single experiment from a config file and print the result.

Usage
~~~~~
    python scripts/run_experiment.py
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/base.yaml --strategy momentum
    python scripts/run_experiment.py --notes "trying larger lookback"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_research.config import load_config, default_config
from trading_research.experiment_runner import run_experiment
from trading_research.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["momentum", "mean_reversion", "ma_crossover"],
        help="Override strategy name",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Data split",
    )
    parser.add_argument("--notes", default="", help="Experiment notes")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    if args.strategy:
        cfg.strategy.name = args.strategy

    print(f"\nRunning experiment | strategy={cfg.strategy.name} | split={args.split}")

    record = run_experiment(cfg, split=args.split, notes=args.notes)

    print(f"\n{'='*55}")
    print(f" Experiment #{record['experiment_id']} Result")
    print(f"{'='*55}")
    print(f" Strategy    : {record['strategy_name']}")
    print(f" Passed      : {record['passed']}")
    print(f" Scalar score: {record.get('scalar_score', 'N/A')}")
    print(f" Data window : {record['data_window']}")

    m = record.get("metrics", {})
    print(f"\n Metrics:")
    print(f"   Total return      : {m.get('total_return', 0):.2%}")
    print(f"   Annualised return : {m.get('annualised_return', 0):.2%}")
    print(f"   Sortino ratio     : {m.get('sortino', 0):.3f}")
    print(f"   Sharpe ratio      : {m.get('sharpe', 0):.3f}")
    print(f"   Max drawdown      : {m.get('max_drawdown', 0):.2%}")
    print(f"   Annual turnover   : {m.get('annualised_turnover', 0):.2f}x")
    print(f"   N trades          : {m.get('n_trades', 0)}")
    print(f"   HHI concentration : {m.get('hhi_concentration', 0):.3f}")

    if record.get("fail_reasons"):
        print(f"\n Failure reasons:")
        for reason in record["fail_reasons"]:
            print(f"   ✗ {reason}")

    if record.get("git_hash"):
        print(f"\n Git hash: {record['git_hash']}")

    print(f"\nLogged to: {cfg.experiment.log_path}")


if __name__ == "__main__":
    main()
