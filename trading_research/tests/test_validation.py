"""
test_validation.py
------------------
Tests for validation.py.

Verifies:
- Train/val/test splits are non-overlapping and chronological
- Fractions sum correctly
- Walk-forward folds are non-overlapping
- Test set is strictly after validation
"""

import pytest
import pandas as pd
import numpy as np

from trading_research.data_loader import generate_sample_data
from trading_research.validation import time_split, walk_forward_folds, DataSplit
from trading_research.config import SplitsConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def five_year_df():
    return generate_sample_data(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2015-01-01",
        end_date="2020-12-31",
        seed=42,
    )


@pytest.fixture
def default_splits_cfg():
    return SplitsConfig(train=0.6, validation=0.2, test=0.2)


# ---------------------------------------------------------------------------
# time_split tests
# ---------------------------------------------------------------------------

def test_split_covers_all_dates(five_year_df, default_splits_cfg):
    split = time_split(five_year_df, default_splits_cfg)
    total_dates = len(five_year_df["date"].unique())
    split_dates = (
        len(split.train["date"].unique())
        + len(split.validation["date"].unique())
        + len(split.test["date"].unique())
    )
    assert split_dates == total_dates


def test_split_no_overlap(five_year_df, default_splits_cfg):
    split = time_split(five_year_df, default_splits_cfg)
    train_dates = set(split.train["date"])
    val_dates = set(split.validation["date"])
    test_dates = set(split.test["date"])

    assert len(train_dates & val_dates) == 0, "Train and validation dates overlap"
    assert len(train_dates & test_dates) == 0, "Train and test dates overlap"
    assert len(val_dates & test_dates) == 0, "Validation and test dates overlap"


def test_split_chronological_order(five_year_df, default_splits_cfg):
    split = time_split(five_year_df, default_splits_cfg)
    assert split.train_end < split.val_start, "Train must end before validation starts"
    assert split.val_end < split.test_start, "Validation must end before test starts"


def test_test_is_most_recent(five_year_df, default_splits_cfg):
    """Test set must contain the most recent dates."""
    split = time_split(five_year_df, default_splits_cfg)
    overall_end = five_year_df["date"].max()
    assert split.test_end == overall_end, "Test set must end at the last date in the dataset"


def test_split_fractions_approximately_correct(five_year_df, default_splits_cfg):
    split = time_split(five_year_df, default_splits_cfg)
    total = len(five_year_df["date"].unique())
    train_frac = len(split.train["date"].unique()) / total
    val_frac = len(split.validation["date"].unique()) / total
    test_frac = len(split.test["date"].unique()) / total

    assert abs(train_frac - 0.6) < 0.05
    assert abs(val_frac - 0.2) < 0.05
    assert abs(test_frac - 0.2) < 0.05


def test_split_invalid_fractions():
    with pytest.raises(Exception):
        SplitsConfig(train=0.5, validation=0.4, test=0.3)  # sums to 1.2


def test_split_single_symbol(default_splits_cfg):
    df = generate_sample_data(["AAPL"], "2015-01-01", "2022-12-31", seed=1)
    split = time_split(df, default_splits_cfg)
    assert not split.train.empty
    assert not split.validation.empty
    assert not split.test.empty


def test_split_too_few_dates(default_splits_cfg):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2),
        "open": [1.0, 2.0],
        "high": [1.0, 2.0],
        "low": [1.0, 2.0],
        "close": [1.0, 2.0],
        "volume": [1000.0, 1000.0],
        "symbol": ["A", "A"],
    })
    with pytest.raises(ValueError, match="at least 3"):
        time_split(df, default_splits_cfg)


# ---------------------------------------------------------------------------
# walk_forward_folds tests
# ---------------------------------------------------------------------------

def test_walk_forward_folds_count(five_year_df):
    cfg = SplitsConfig(
        walkforward_train_years=2,
        walkforward_val_years=1,
        walkforward_step_months=12,
    )
    folds = walk_forward_folds(five_year_df, cfg)
    assert len(folds) >= 1, "Should produce at least one fold"


def test_walk_forward_folds_no_overlap(five_year_df):
    cfg = SplitsConfig(
        walkforward_train_years=1,
        walkforward_val_years=1,
        walkforward_step_months=6,
    )
    folds = walk_forward_folds(five_year_df, cfg)
    for fold in folds:
        train_dates = set(fold.train["date"])
        val_dates = set(fold.validation["date"])
        assert len(train_dates & val_dates) == 0, (
            f"Fold {fold.fold_id}: train and validation dates overlap"
        )


def test_walk_forward_folds_chronological(five_year_df):
    cfg = SplitsConfig(
        walkforward_train_years=1,
        walkforward_val_years=1,
        walkforward_step_months=6,
    )
    folds = walk_forward_folds(five_year_df, cfg)
    for fold in folds:
        assert fold.train_end < fold.val_start, (
            f"Fold {fold.fold_id}: train must precede validation"
        )


def test_walk_forward_empty_on_tiny_data():
    df = generate_sample_data(["AAPL"], "2020-01-01", "2020-06-30", seed=0)
    cfg = SplitsConfig(
        walkforward_train_years=2,
        walkforward_val_years=1,
        walkforward_step_months=6,
    )
    folds = walk_forward_folds(df, cfg)
    assert folds == [], "Should return empty list when insufficient data"
