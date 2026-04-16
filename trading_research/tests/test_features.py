"""
test_features.py
----------------
Tests for features.py, with special attention to no-lookahead guarantees.
"""

import pytest
import pandas as pd
import numpy as np

from trading_research.data_loader import generate_sample_data, pivot_ohlcv
from trading_research.features import (
    rolling_return,
    rolling_volatility,
    moving_average,
    ma_ratio,
    volume_ratio,
    z_score,
    build_feature_matrix,
    build_feature_matrix_from_df,
)
from trading_research.config import FeaturesConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def close_wide():
    df = generate_sample_data(["AAPL", "MSFT"], "2020-01-01", "2022-12-31", seed=7)
    return pivot_ohlcv(df, "close")


@pytest.fixture
def volume_wide():
    df = generate_sample_data(["AAPL", "MSFT"], "2020-01-01", "2022-12-31", seed=7)
    return pivot_ohlcv(df, "volume")


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------

def test_rolling_return_no_lookahead(close_wide):
    """rolling_return at date t must only use data up to and including t."""
    ret = rolling_return(close_wide, window=5)
    # By construction, pct_change(5) at index i uses rows i-5..i: no lookahead.
    # Verify: if we remove the last row and recompute, values for earlier rows are unchanged.
    ret_trimmed = rolling_return(close_wide.iloc[:-1], window=5)
    pd.testing.assert_frame_equal(
        ret.iloc[:-1],
        ret_trimmed,
        check_names=False,
    )


def test_moving_average_no_lookahead(close_wide):
    """MA at t depends only on t and prior rows."""
    ma = moving_average(close_wide, window=20)
    ma_trimmed = moving_average(close_wide.iloc[:-1], window=20)
    pd.testing.assert_frame_equal(ma.iloc[:-1], ma_trimmed, check_names=False)


def test_z_score_no_lookahead(close_wide):
    """z_score at t uses only rows up to t."""
    ret = rolling_return(close_wide, window=20)
    zs = z_score(ret, window=60)
    zs_trimmed = z_score(rolling_return(close_wide.iloc[:-1], window=20), window=60)
    pd.testing.assert_frame_equal(zs.iloc[:-1], zs_trimmed, check_names=False)


# ---------------------------------------------------------------------------
# Shape and NaN tests
# ---------------------------------------------------------------------------

def test_rolling_return_shape(close_wide):
    ret = rolling_return(close_wide, window=5)
    assert ret.shape == close_wide.shape


def test_rolling_volatility_shape(close_wide):
    vol = rolling_volatility(close_wide, window=20)
    assert vol.shape == close_wide.shape


def test_moving_average_shape(close_wide):
    ma = moving_average(close_wide, window=50)
    assert ma.shape == close_wide.shape


def test_rolling_return_nans_at_start(close_wide):
    ret = rolling_return(close_wide, window=10)
    # First 10 rows should be NaN (no prior data for 10-day return)
    assert ret.iloc[:10].isna().all(axis=None)
    # After warmup, values should be present
    assert not ret.iloc[11:].isna().all(axis=None)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_rolling_return_manual():
    """Spot-check rolling_return against manual calculation."""
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1]},
        index=pd.date_range("2020-01-01", periods=4),
    )
    ret = rolling_return(prices, window=1)
    expected = pd.DataFrame(
        {"A": [np.nan, 0.1, 0.1, 0.1]},
        index=prices.index,
    )
    pd.testing.assert_frame_equal(ret, expected)


def test_moving_average_manual():
    prices = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=5),
    )
    ma = moving_average(prices, window=3)
    # row 2 = mean(1,2,3)=2, row 3=mean(2,3,4)=3, row 4=mean(3,4,5)=4
    assert abs(ma.iloc[2]["A"] - 2.0) < 1e-9
    assert abs(ma.iloc[3]["A"] - 3.0) < 1e-9
    assert abs(ma.iloc[4]["A"] - 4.0) < 1e-9


def test_build_feature_matrix_keys(close_wide, volume_wide):
    cfg = FeaturesConfig(return_windows=[5, 20], volatility_window=20, ma_windows=[20, 50])
    features = build_feature_matrix(close_wide, volume_wide, cfg)
    assert "return_5d" in features
    assert "return_20d" in features
    assert "volatility_20d" in features
    assert "ma_20d" in features
    assert "ma_50d" in features


def test_build_feature_matrix_shapes(close_wide, volume_wide):
    cfg = FeaturesConfig()
    features = build_feature_matrix(close_wide, volume_wide, cfg)
    for name, df in features.items():
        assert df.shape == close_wide.shape, f"Shape mismatch for feature '{name}'"
