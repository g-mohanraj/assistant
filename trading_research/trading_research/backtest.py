"""
backtest.py
-----------
Daily-bar backtester.

Execution model
~~~~~~~~~~~~~~~
- Signals are generated at the CLOSE of day t using data available by t.
- Positions are opened at the OPEN of day t+1 (next-bar execution).
- This prevents lookahead bias from same-bar fills.

Transaction costs
~~~~~~~~~~~~~~~~~
- Round-trip cost = (transaction_cost_bps + slippage_bps) * 2 / 10_000
- Applied to the absolute change in position value on each rebalance.

Outputs
~~~~~~~
- equity_curve : pd.Series — portfolio value each day
- returns      : pd.Series — daily portfolio returns
- positions    : pd.DataFrame — dollar positions per symbol
- weights      : pd.DataFrame — weight per symbol after normalisation
- trades       : pd.DataFrame — trade-level log
- drawdown     : pd.Series — drawdown from peak at each date
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_research.config import BacktestConfig

logger = logging.getLogger(__name__)

BPS = 1e-4  # one basis point


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """All outputs from a single backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame        # dollar positions per symbol
    weights: pd.DataFrame          # weight per symbol
    trades: pd.DataFrame           # trade log
    drawdown: pd.Series

    # Convenience properties ------------------------------------------------

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1)

    @property
    def n_trading_days(self) -> int:
        return len(self.equity_curve)

    @property
    def annualised_return(self) -> float:
        if self.n_trading_days < 2:
            return 0.0
        years = self.n_trading_days / 252.0
        return float((1 + self.total_return) ** (1.0 / years) - 1)

    @property
    def max_drawdown(self) -> float:
        return float(self.drawdown.min()) if not self.drawdown.empty else 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def annualised_turnover(self) -> float:
        """Annualised one-way turnover as a fraction of portfolio value."""
        if self.n_trading_days < 2:
            return 0.0
        total_turnover = self.trades["turnover_fraction"].sum() if not self.trades.empty else 0.0
        return float(total_turnover * 252 / self.n_trading_days)

    def summary(self) -> dict:
        return {
            "total_return": round(self.total_return, 6),
            "annualised_return": round(self.annualised_return, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "n_trades": self.n_trades,
            "annualised_turnover": round(self.annualised_turnover, 6),
            "n_trading_days": self.n_trading_days,
        }


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    open_prices: Optional[pd.DataFrame] = None,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorised daily-bar backtest.

    Parameters
    ----------
    weights:
        Target weight DataFrame (date x symbol) output by the portfolio layer.
        Row at date t means: "at the close of t, target these weights."
        Execution happens at the open of t+1.
    close:
        Wide close-price DataFrame (date x symbol). Used for mark-to-market.
    open_prices:
        Wide open-price DataFrame (date x symbol). Used for execution fills.
        If None, uses close prices as a conservative proxy.
    cfg:
        BacktestConfig. Uses defaults if None.

    Returns
    -------
    BacktestResult
    """
    if cfg is None:
        cfg = BacktestConfig()

    # Align weights and prices on common dates and symbols
    common_dates = weights.index.intersection(close.index)
    common_syms = weights.columns.intersection(close.columns)

    weights = weights.loc[common_dates, common_syms]
    close = close.loc[common_dates, common_syms]

    if open_prices is not None:
        open_prices = open_prices.reindex(index=common_dates, columns=common_syms)
    else:
        open_prices = close.copy()

    if weights.empty or close.empty:
        return _empty_result(cfg)

    # Shift weights by 1 day: signal from close of t executes at open of t+1
    # weights_shifted[t] = the weight we execute at open_prices[t]
    weights_shifted = weights.shift(1).fillna(0.0)

    n_dates = len(common_dates)
    initial_capital = cfg.initial_capital
    one_way_cost = (cfg.transaction_cost_bps + cfg.slippage_bps) * BPS

    equity = np.empty(n_dates)
    equity[0] = initial_capital

    # Dollar positions: what we hold going into each day's close
    dollar_positions = pd.DataFrame(
        0.0, index=common_dates, columns=common_syms
    )
    trade_records = []

    current_shares = pd.Series(0.0, index=common_syms)
    portfolio_value = initial_capital

    for i in range(n_dates):
        date = common_dates[i]

        if i == 0:
            # First day: no prior positions; just record initial equity
            equity[0] = portfolio_value
            dollar_positions.iloc[0] = 0.0
            continue

        # Execution price = open of this bar
        exec_prices = open_prices.iloc[i]

        # Target weights were set at previous close
        target_weights = weights_shifted.iloc[i]

        # Target dollar amounts
        target_dollars = target_weights * portfolio_value
        # Target shares
        target_shares = target_dollars / exec_prices.replace(0, np.nan).fillna(np.inf)
        target_shares = target_shares.fillna(0.0)

        # Trade = change in shares
        delta_shares = target_shares - current_shares
        traded_symbols = delta_shares[delta_shares.abs() > 1e-9]

        # Compute transaction costs on traded notional
        traded_notional = (delta_shares.abs() * exec_prices).sum()
        cost = traded_notional * one_way_cost

        # Update shares
        current_shares = target_shares.copy()

        # Mark to market at close of this bar
        close_prices = close.iloc[i]
        position_value = (current_shares * close_prices).sum()
        cash = portfolio_value - (target_shares * exec_prices).sum() - cost
        portfolio_value = position_value + cash
        portfolio_value = max(portfolio_value, 0.0)

        equity[i] = portfolio_value

        # Record positions
        dollar_positions.iloc[i] = current_shares * close_prices

        # Record trades
        if not traded_symbols.empty:
            turnover_fraction = traded_notional / max(portfolio_value, 1.0)
            for sym, dshares in traded_symbols.items():
                trade_records.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "delta_shares": float(dshares),
                        "exec_price": float(exec_prices[sym]),
                        "turnover_fraction": float(
                            abs(dshares) * exec_prices[sym] / max(portfolio_value, 1.0)
                        ),
                        "cost": float(abs(dshares) * exec_prices[sym] * one_way_cost),
                    }
                )

    equity_series = pd.Series(equity, index=common_dates, name="equity")
    returns = equity_series.pct_change().fillna(0.0)
    drawdown = _compute_drawdown(equity_series)

    trades_df = pd.DataFrame(trade_records) if trade_records else _empty_trades()

    return BacktestResult(
        equity_curve=equity_series,
        returns=returns,
        positions=dollar_positions,
        weights=weights_shifted,
        trades=trades_df,
        drawdown=drawdown,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown from running peak."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    return dd.fillna(0.0)


def _empty_result(cfg: BacktestConfig) -> BacktestResult:
    empty = pd.Series(dtype=float)
    empty_df = pd.DataFrame()
    return BacktestResult(
        equity_curve=empty,
        returns=empty,
        positions=empty_df,
        weights=empty_df,
        trades=_empty_trades(),
        drawdown=empty,
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "delta_shares", "exec_price", "turnover_fraction", "cost"]
    )
