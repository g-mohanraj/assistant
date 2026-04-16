"""
test_backtest.py
----------------
Tests for backtest.py.

Key invariants verified:
- Output shape matches input
- Deterministic output (same input -> same output)
- No lookahead (signal shift-by-1 is applied)
- All-cash strategy produces flat equity curve (minus costs)
- Equity curve starts at initial_capital
"""

import pytest
import numpy as np
import pandas as pd

from trading_research.backtest import run_backtest, BacktestResult
from trading_research.config import BacktestConfig
from trading_research.data_loader import generate_sample_data, pivot_ohlcv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_data():
    df = generate_sample_data(
        symbols=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2021-12-31",
        seed=42,
    )
    close = pivot_ohlcv(df, "close")
    open_ = pivot_ohlcv(df, "open")
    return close, open_


@pytest.fixture
def zero_cost_cfg():
    return BacktestConfig(
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        initial_capital=100_000.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_equity_starts_at_initial_capital(price_data, zero_cost_cfg):
    close, open_ = price_data
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    result = run_backtest(weights, close, open_, zero_cost_cfg)
    assert abs(result.equity_curve.iloc[0] - zero_cost_cfg.initial_capital) < 1.0


def test_all_cash_flat_equity(price_data, zero_cost_cfg):
    """All-zero weights should produce flat equity (no costs, no positions)."""
    close, open_ = price_data
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    result = run_backtest(weights, close, open_, zero_cost_cfg)
    # Equity should remain at initial_capital throughout
    assert result.equity_curve.std() < 1.0, "All-cash equity should be flat"


def test_output_shape(price_data, zero_cost_cfg):
    close, open_ = price_data
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    weights = weights.div(weights.sum(axis=1), axis=0)  # normalise
    result = run_backtest(weights, close, open_, zero_cost_cfg)
    assert len(result.equity_curve) == len(close)
    assert len(result.returns) == len(close)
    assert result.positions.shape == close.shape


def test_deterministic(price_data):
    close, open_ = price_data
    cfg = BacktestConfig()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    weights = weights.div(weights.sum(axis=1), axis=0)

    result1 = run_backtest(weights, close, open_, cfg)
    result2 = run_backtest(weights, close, open_, cfg)

    pd.testing.assert_series_equal(result1.equity_curve, result2.equity_curve)
    pd.testing.assert_series_equal(result1.returns, result2.returns)


def test_transaction_costs_reduce_equity(price_data):
    """Backtests with costs should underperform the zero-cost version."""
    close, open_ = price_data
    # Create a strategy that trades every day (maximum turnover)
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for i, date in enumerate(close.index):
        if i % 2 == 0:
            weights.loc[date, "AAPL"] = 1.0
            weights.loc[date, "MSFT"] = 0.0
        else:
            weights.loc[date, "AAPL"] = 0.0
            weights.loc[date, "MSFT"] = 1.0

    no_cost = BacktestConfig(transaction_cost_bps=0, slippage_bps=0)
    with_cost = BacktestConfig(transaction_cost_bps=20, slippage_bps=10)

    r_no_cost = run_backtest(weights, close, open_, no_cost)
    r_with_cost = run_backtest(weights, close, open_, with_cost)

    # Final equity should be lower with costs
    assert r_with_cost.equity_curve.iloc[-1] < r_no_cost.equity_curve.iloc[-1]


def test_no_lookahead_in_execution(price_data, zero_cost_cfg):
    """Signal from day t must not affect equity until day t+1."""
    close, open_ = price_data
    n = len(close)

    # Strategy: buy AAPL on day 10, nothing before
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights.iloc[10, weights.columns.get_loc("AAPL")] = 1.0

    result = run_backtest(weights, close, open_, zero_cost_cfg)

    # Equity should be flat for first 11 days (signal on day 10 executes at day 11 open)
    assert result.equity_curve.iloc[:11].std() < 1.0, (
        "Position should not open until day 11 (next-bar execution)"
    )


def test_drawdown_is_nonpositive(price_data):
    close, open_ = price_data
    cfg = BacktestConfig()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    weights = weights.div(weights.sum(axis=1), axis=0)
    result = run_backtest(weights, close, open_, cfg)
    assert (result.drawdown <= 0.0).all(), "Drawdown must always be <= 0"


def test_returns_length_matches_equity(price_data):
    close, open_ = price_data
    cfg = BacktestConfig()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    weights = weights.div(weights.sum(axis=1), axis=0)
    result = run_backtest(weights, close, open_, cfg)
    assert len(result.returns) == len(result.equity_curve)


def test_trade_log_columns(price_data):
    close, open_ = price_data
    cfg = BacktestConfig()
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    weights = weights.div(weights.sum(axis=1), axis=0)
    result = run_backtest(weights, close, open_, cfg)
    if not result.trades.empty:
        expected_cols = {"date", "symbol", "delta_shares", "exec_price", "turnover_fraction", "cost"}
        assert expected_cols.issubset(set(result.trades.columns))
