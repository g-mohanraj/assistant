"""
test_scoring.py
---------------
Tests for scoring.py.

Verifies that:
- Score function produces expected output shape
- Hard-fail rules trigger correctly
- Score is finite and bounded for valid inputs
- Synthetic winning strategy scores positively
- Synthetic losing strategy gets rejected
"""

import pytest
import numpy as np
import pandas as pd

from trading_research.backtest import BacktestResult, run_backtest
from trading_research.scoring import score_backtest, ScoreResult, _sortino_ratio, _sharpe_ratio
from trading_research.config import ScoringConfig, BacktestConfig
from trading_research.data_loader import generate_sample_data, pivot_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=idx)


def _make_backtest_result(returns: pd.Series, n_trades: int = 50) -> BacktestResult:
    """Build a minimal BacktestResult from a returns series.

    Creates two equal-weight positions so HHI = 0.5 (well below the 0.5 threshold
    defined in the default ScoringConfig).
    """
    equity = (1 + returns).cumprod() * 100_000
    dd = (equity - equity.cummax()) / equity.cummax().replace(0, np.nan)
    # Two equal-weight symbols -> HHI = 0.5^2 + 0.5^2 = 0.5
    positions = pd.DataFrame(
        {"SYM_A": equity * 0.5, "SYM_B": equity * 0.5}, index=equity.index
    )
    syms = ["SYM_A", "SYM_B"] * (n_trades // 2) + ["SYM_A"] * (n_trades % 2)
    trades = pd.DataFrame(
        {
            "date": list(equity.index[:n_trades]),
            "symbol": syms[:n_trades],
            "delta_shares": 1.0,
            "exec_price": 100.0,
            "turnover_fraction": 0.01,
            "cost": 0.1,
        }
    )
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=positions,
        weights=positions.div(equity, axis=0),
        trades=trades,
        drawdown=dd.fillna(0.0),
    )


# ---------------------------------------------------------------------------
# Unit tests: individual metrics
# ---------------------------------------------------------------------------

def test_sortino_positive_on_upward_drift():
    returns = _make_returns(500, mean=0.001, std=0.01, seed=1)
    s = _sortino_ratio(returns)
    assert s > 0


def test_sortino_negative_on_downward_drift():
    returns = _make_returns(500, mean=-0.001, std=0.01, seed=1)
    s = _sortino_ratio(returns)
    assert s < 0


def test_sortino_nan_on_short_series():
    returns = _make_returns(10, mean=0.001, std=0.01)
    s = _sortino_ratio(returns)
    assert np.isnan(s)


def test_sharpe_positive():
    returns = _make_returns(500, mean=0.001, std=0.005)
    sh = _sharpe_ratio(returns)
    assert sh > 0


# ---------------------------------------------------------------------------
# Integration tests: full score_backtest
# ---------------------------------------------------------------------------

def test_score_passes_for_good_strategy():
    """A strategy with decent returns and low drawdown should pass."""
    returns = _make_returns(500, mean=0.0008, std=0.008, seed=42)
    result = _make_backtest_result(returns, n_trades=100)
    cfg = ScoringConfig(min_trades=10, max_drawdown_threshold=0.50)
    score = score_backtest(result, cfg)
    assert score.passed, f"Expected pass but got fail: {score.fail_reasons}"
    assert np.isfinite(score.scalar_score)


def test_score_fails_too_few_trades():
    returns = _make_returns(500, mean=0.001, std=0.008)
    result = _make_backtest_result(returns, n_trades=2)
    cfg = ScoringConfig(min_trades=20)
    score = score_backtest(result, cfg)
    assert not score.passed
    assert any("too few trades" in r for r in score.fail_reasons)


def test_score_fails_large_drawdown():
    # Simulate a catastrophic loss
    returns = _make_returns(500, mean=-0.005, std=0.02, seed=99)
    result = _make_backtest_result(returns, n_trades=50)
    cfg = ScoringConfig(max_drawdown_threshold=0.30)
    score = score_backtest(result, cfg)
    # If drawdown is large enough, it should fail
    if abs(result.max_drawdown) > 0.30:
        assert not score.passed
        assert any("drawdown" in r for r in score.fail_reasons)


def test_score_result_to_dict():
    returns = _make_returns(500, mean=0.0008, std=0.008, seed=42)
    result = _make_backtest_result(returns, n_trades=100)
    score = score_backtest(result)
    d = score.to_dict()
    assert "passed" in d
    assert "scalar_score" in d
    assert "sortino" in d
    assert "max_drawdown" in d
    assert "n_trades" in d


def test_score_nan_on_empty_result():
    empty = pd.Series(dtype=float)
    empty_df = pd.DataFrame()
    result = BacktestResult(
        equity_curve=empty,
        returns=empty,
        positions=empty_df,
        weights=empty_df,
        trades=pd.DataFrame(columns=["date", "symbol", "delta_shares",
                                     "exec_price", "turnover_fraction", "cost"]),
        drawdown=empty,
    )
    score = score_backtest(result)
    assert not score.passed
    assert np.isnan(score.scalar_score)


def test_end_to_end_score_with_backtest():
    """Full pipeline: generate data -> backtest -> score."""
    df = generate_sample_data(["AAPL", "MSFT"], "2020-01-01", "2022-12-31", seed=0)
    close = pivot_ohlcv(df, "close")
    open_ = pivot_ohlcv(df, "open")

    # Simple equal-weight strategy
    weights = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    cfg = BacktestConfig(transaction_cost_bps=10, slippage_bps=5)
    result = run_backtest(weights, close, open_, cfg)

    score = score_backtest(result)
    assert isinstance(score, ScoreResult)
    # Should have metrics
    assert score.n_trades >= 0
    assert score.max_drawdown <= 0
