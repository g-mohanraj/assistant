"""
validation.py
-------------
Train / validation / test splits and walk-forward validation utilities.

Data hygiene guarantee
~~~~~~~~~~~~~~~~~~~~~~
The test set is NEVER accessible to the strategy search loop.
The experiment runner and walk-forward validator operate only on
train + validation splits. The test set is reserved for final evaluation.

Walk-forward
~~~~~~~~~~~~
The walk-forward validator slides a training window forward in time,
testing each fold on the next out-of-sample period. This avoids
optimistic in-sample bias and gives a realistic picture of how the
strategy performs on unseen data.

Typical usage
~~~~~~~~~~~~~
    splits = time_split(df, cfg.splits)
    train_df = splits["train"]
    val_df   = splits["validation"]
    # ... search loop on train_df + val_df ...
    # Only AFTER settling on a strategy:
    test_df  = splits["test"]
    result   = evaluate_on_test(best_strategy, test_df, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from trading_research.config import SplitsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    """Named date-range slices of the full dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train["date"].min()

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train["date"].max()

    @property
    def val_start(self) -> pd.Timestamp:
        return self.validation["date"].min()

    @property
    def val_end(self) -> pd.Timestamp:
        return self.validation["date"].max()

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test["date"].min()

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test["date"].max()

    def summary(self) -> dict:
        return {
            "train": f"{self.train_start.date()} → {self.train_end.date()} "
                     f"({len(self.train['date'].unique())} days)",
            "validation": f"{self.val_start.date()} → {self.val_end.date()} "
                          f"({len(self.validation['date'].unique())} days)",
            "test": f"{self.test_start.date()} → {self.test_end.date()} "
                    f"({len(self.test['date'].unique())} days)",
        }


@dataclass
class WalkForwardFold:
    """A single walk-forward fold."""

    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train["date"].min()

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train["date"].max()

    @property
    def val_start(self) -> pd.Timestamp:
        return self.validation["date"].min()

    @property
    def val_end(self) -> pd.Timestamp:
        return self.validation["date"].max()


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    cfg: SplitsConfig | None = None,
) -> DataSplit:
    """Split a long-format OHLCV DataFrame into train/validation/test sets.

    Splits are chronological (no shuffle). The test set contains the most
    recent data, then validation, then train.

    Parameters
    ----------
    df:
        Long-format DataFrame with a ``date`` column.
    cfg:
        SplitsConfig. Uses defaults if None.

    Returns
    -------
    DataSplit
        Three non-overlapping subsets covering the full date range.
    """
    if cfg is None:
        cfg = SplitsConfig()

    dates = df["date"].sort_values().unique()
    n = len(dates)
    if n < 3:
        raise ValueError(f"Need at least 3 unique dates to split; got {n}.")

    train_end_idx = int(n * cfg.train)
    val_end_idx = int(n * (cfg.train + cfg.validation))

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    split = DataSplit(train=train_df, validation=val_df, test=test_df)
    logger.info("Data split: %s", split.summary())
    return split


def get_train_val(
    df: pd.DataFrame,
    cfg: SplitsConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return only the train and validation portions (no test leakage).

    Use this inside the strategy search loop. Never pass test data here.
    """
    split = time_split(df, cfg)
    return split.train, split.validation


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_folds(
    df: pd.DataFrame,
    cfg: SplitsConfig | None = None,
) -> List[WalkForwardFold]:
    """Generate a list of walk-forward folds from the TRAIN+VAL portion only.

    Each fold has:
      - A rolling training window of ``train_years`` years.
      - A validation window of ``val_years`` years immediately after.
      - The window slides forward by ``step_months`` months per fold.

    Parameters
    ----------
    df:
        Long-format DataFrame (train+val portion — do NOT pass test data).
    cfg:
        SplitsConfig.

    Returns
    -------
    list[WalkForwardFold]
    """
    if cfg is None:
        cfg = SplitsConfig()

    dates = df["date"].sort_values().unique()
    if len(dates) == 0:
        return []

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])

    train_delta = pd.DateOffset(years=cfg.walkforward_train_years)
    val_delta = pd.DateOffset(years=cfg.walkforward_val_years)
    step_delta = pd.DateOffset(months=cfg.walkforward_step_months)

    folds = []
    fold_id = 0
    cursor = start

    while True:
        train_start = cursor
        train_end = cursor + train_delta
        val_start = train_end
        val_end = val_start + val_delta

        if val_end > end:
            break

        train_mask = (df["date"] >= train_start) & (df["date"] < train_end)
        val_mask = (df["date"] >= val_start) & (df["date"] < val_end)

        train_slice = df[train_mask].copy()
        val_slice = df[val_mask].copy()

        if train_slice.empty or val_slice.empty:
            cursor += step_delta
            continue

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train=train_slice,
                validation=val_slice,
            )
        )
        fold_id += 1
        cursor += step_delta

    logger.info("Generated %d walk-forward folds.", len(folds))
    return folds


# ---------------------------------------------------------------------------
# Final test evaluation guard
# ---------------------------------------------------------------------------

def evaluate_on_test(
    test_df: pd.DataFrame,
    strategy,
    portfolio_cfg=None,
    backtest_cfg=None,
    scoring_cfg=None,
    features_cfg=None,
) -> dict:
    """Run a FINAL evaluation on held-out test data.

    This function should only be called ONCE, after the strategy search is
    completely done on train/validation data. Calling it multiple times
    for search purposes defeats the purpose of a holdout set.

    Parameters
    ----------
    test_df:
        Held-out test split from time_split.
    strategy:
        A StrategyBase instance (fully configured).
    portfolio_cfg, backtest_cfg, scoring_cfg, features_cfg:
        Optional config overrides.

    Returns
    -------
    dict
        Scoring metrics on the test set.
    """
    from trading_research.features import build_feature_matrix_from_df
    from trading_research.data_loader import pivot_ohlcv
    from trading_research.portfolio import compute_weights_vectorised
    from trading_research.backtest import run_backtest
    from trading_research.scoring import score_backtest

    close = pivot_ohlcv(test_df, "close")
    features = build_feature_matrix_from_df(test_df, features_cfg)
    signals = strategy.generate_signals(close, features)
    weights = compute_weights_vectorised(signals, portfolio_cfg)
    open_prices = pivot_ohlcv(test_df, "open")
    bt_result = run_backtest(weights, close, open_prices, backtest_cfg)
    score = score_backtest(bt_result, scoring_cfg)

    return {"test_" + k: v for k, v in score.to_dict().items()}
