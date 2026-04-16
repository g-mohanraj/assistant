"""
experiment_runner.py
--------------------
Orchestrates a single end-to-end experiment:
  data → features → strategy → portfolio → backtest → scoring → log

Usage
~~~~~
    from trading_research.config import load_config
    from trading_research.experiment_runner import run_experiment

    cfg = load_config("configs/base.yaml")
    record = run_experiment(cfg, split="validation", notes="baseline run")

The runner always operates on the train+validation window.
Pass split="validation" to backtest on the validation fold.
Pass split="train" to backtest on the training fold (use for sanity checks only).

WARNING: Do not pass split="test" here — use validation.evaluate_on_test instead.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import pandas as pd

from trading_research.config import Config, load_config
from trading_research.data_loader import load_from_config, pivot_ohlcv
from trading_research.features import build_feature_matrix
from trading_research.strategy import build_strategy
from trading_research.portfolio import compute_weights_vectorised
from trading_research.backtest import run_backtest
from trading_research.scoring import score_backtest
from trading_research.validation import time_split
from trading_research.experiment_logger import log_experiment

logger = logging.getLogger(__name__)

SplitName = Literal["train", "validation"]


def run_experiment(
    cfg: Config,
    split: SplitName = "validation",
    notes: str = "",
) -> dict:
    """Run a full experiment and append the result to the experiment log.

    Parameters
    ----------
    cfg:
        Fully resolved Config object.
    split:
        Which data split to evaluate on. Must be "train" or "validation".
        Never pass "test" — that split is reserved for final evaluation only.
    notes:
        Free-text rationale or description (written into the log).

    Returns
    -------
    dict
        The experiment record that was written to the log.
    """
    logger.info(
        "Starting experiment | strategy=%s | split=%s",
        cfg.strategy.name,
        split,
    )

    # ---- 1. Load data ----
    df = load_from_config(cfg)
    splits = time_split(df, cfg.splits)

    if split == "train":
        eval_df = splits.train
    elif split == "validation":
        eval_df = splits.validation
    else:
        raise ValueError(
            f"split must be 'train' or 'validation', not '{split}'. "
            "Use validation.evaluate_on_test() for test evaluation."
        )

    if eval_df.empty:
        raise ValueError(f"The '{split}' split produced an empty DataFrame.")

    # ---- 2. Build features ----
    close = pivot_ohlcv(eval_df, "close")
    volume = pivot_ohlcv(eval_df, "volume")
    open_prices = pivot_ohlcv(eval_df, "open")

    features = build_feature_matrix(close, volume, cfg.features)

    # ---- 3. Generate signals ----
    strategy = build_strategy(cfg.strategy)
    signals = strategy.generate_signals(close, features)
    strategy.validate_signals(signals, close)

    # ---- 4. Compute portfolio weights ----
    weights = compute_weights_vectorised(signals, cfg.portfolio)

    # ---- 5. Backtest ----
    bt_result = run_backtest(weights, close, open_prices, cfg.backtest)

    logger.info(
        "Backtest complete | return=%.2f%% | max_dd=%.2f%% | n_trades=%d",
        bt_result.total_return * 100,
        bt_result.max_drawdown * 100,
        bt_result.n_trades,
    )

    # ---- 6. Score ----
    score_result = score_backtest(bt_result, cfg.scoring)

    logger.info(
        "Score | passed=%s | scalar=%.4f | sortino=%.2f",
        score_result.passed,
        score_result.scalar_score if score_result.passed else float("nan"),
        score_result.sortino,
    )

    # ---- 7. Log ----
    data_window = {
        "start": str(eval_df["date"].min().date()),
        "end": str(eval_df["date"].max().date()),
        "split": split,
        "n_symbols": int(eval_df["symbol"].nunique()),
    }

    config_snapshot = cfg.model_dump()

    record = log_experiment(
        log_path=cfg.experiment.log_path,
        strategy_name=cfg.strategy.name,
        config=config_snapshot,
        data_window=data_window,
        metrics=score_result.to_dict(),
        scalar_score=score_result.scalar_score if score_result.passed else None,
        passed=score_result.passed,
        fail_reasons=score_result.fail_reasons,
        notes=notes or cfg.experiment.notes,
    )

    return record


def run_multiple_strategies(
    cfg: Config,
    strategies: list,
    split: SplitName = "validation",
    notes: str = "",
) -> list:
    """Run the same pipeline for multiple strategy names and return all records.

    Parameters
    ----------
    cfg:
        Base Config. The strategy name is overridden for each run.
    strategies:
        List of strategy name strings, e.g. ["momentum", "mean_reversion"].
    split:
        "train" or "validation".
    notes:
        Notes appended to each record.

    Returns
    -------
    list[dict]
        Experiment records for each strategy.
    """
    import copy

    records = []
    for name in strategies:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.strategy.name = name
        try:
            record = run_experiment(cfg_copy, split=split, notes=notes)
            records.append(record)
        except Exception as exc:
            logger.error("Strategy '%s' failed: %s", name, exc)
    return records
