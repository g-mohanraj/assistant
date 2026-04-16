"""
test_data_loader.py
-------------------
Tests for data_loader.py.
"""

import io
import pytest
import pandas as pd
import numpy as np

from trading_research.data_loader import (
    generate_sample_data,
    _validate_and_clean,
    pivot_close,
    pivot_ohlcv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return generate_sample_data(
        symbols=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2021-12-31",
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_sample_data_shape(sample_df):
    symbols = sample_df["symbol"].unique()
    assert set(symbols) == {"AAPL", "MSFT"}
    assert len(sample_df) > 0


def test_generate_sample_data_columns(sample_df):
    expected = {"date", "open", "high", "low", "close", "volume", "symbol"}
    assert expected.issubset(set(sample_df.columns))


def test_generate_sample_data_no_nans(sample_df):
    for col in ["open", "high", "low", "close", "volume"]:
        assert sample_df[col].isna().sum() == 0, f"NaN in column {col}"


def test_generate_sample_data_ohlc_consistency(sample_df):
    assert (sample_df["high"] >= sample_df["close"]).all()
    assert (sample_df["high"] >= sample_df["open"]).all()
    assert (sample_df["low"] <= sample_df["close"]).all()
    assert (sample_df["low"] <= sample_df["open"]).all()


def test_generate_sample_data_sorted(sample_df):
    dates = sample_df["date"].values
    syms = sample_df["symbol"].values
    # Sort should be stable: first by date, then by symbol
    for i in range(len(sample_df) - 1):
        assert (dates[i], syms[i]) <= (dates[i + 1], syms[i + 1])


def test_generate_sample_data_deterministic():
    df1 = generate_sample_data(["AAPL"], seed=99)
    df2 = generate_sample_data(["AAPL"], seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_sample_data_different_seeds():
    df1 = generate_sample_data(["AAPL"], seed=1)
    df2 = generate_sample_data(["AAPL"], seed=2)
    # Close prices should differ
    assert not (df1["close"].values == df2["close"].values).all()


def test_validate_and_clean_missing_column():
    df = pd.DataFrame({"date": ["2020-01-01"], "open": [100]})  # missing many cols
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_and_clean(df)


def test_validate_and_clean_normalises_columns():
    df = pd.DataFrame({
        "Date": ["2020-01-01"],
        "Open": [100.0],
        "High": [102.0],
        "Low": [99.0],
        "Close": [101.0],
        "Volume": [1_000_000],
        "Symbol": ["AAPL"],
    })
    result = _validate_and_clean(df)
    assert set(result.columns) == {"date", "open", "high", "low", "close", "volume", "symbol"}


def test_pivot_close_shape(sample_df):
    wide = pivot_close(sample_df)
    assert wide.shape[1] == 2  # 2 symbols
    assert wide.index.name == "date"
    assert set(wide.columns) == {"AAPL", "MSFT"}


def test_pivot_ohlcv_valid_column(sample_df):
    wide = pivot_ohlcv(sample_df, "volume")
    assert wide.shape[1] == 2


def test_pivot_ohlcv_invalid_column(sample_df):
    with pytest.raises(ValueError):
        pivot_ohlcv(sample_df, "notacolumn")
