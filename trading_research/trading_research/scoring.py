"""
scoring.py
----------
Strict, risk-aware scoring function for backtested strategies.

Design principle: optimising the scalar score should NOT lead to
strategies that merely overfit or take excessive risk. The score
penalises:
  - High drawdown
  - High turnover (cost-unaware churning)
  - Insufficient trade count (data snooping with few bets)
  - Highly concentrated exposures (single-symbol dominance)

Hard-fail rules cause immediate rejection before any score is computed.
This makes it impossible to game the ranking by cherry-picking edge cases.

This file is NOT agent-editable (by convention). A future agent should
propose strategies and configs, but not change what constitutes a "good"
strategy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from trading_research.backtest import BacktestResult
from trading_research.config import ScoringConfig

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Output of the scoring function."""

    passed: bool                   # True if no hard-fail triggered
    scalar_score: float            # Higher is better; NaN on hard fail
    fail_reasons: list             # List of strings explaining any failures

    # Component metrics
    total_return: float = 0.0
    annualised_return: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    annualised_turnover: float = 0.0
    n_trades: int = 0
    hhi_concentration: float = 0.0  # Herfindahl-Hirschman Index across symbols
    sharpe: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "scalar_score": self.scalar_score,
            "fail_reasons": self.fail_reasons,
            "total_return": round(self.total_return, 6),
            "annualised_return": round(self.annualised_return, 6),
            "sortino": round(self.sortino, 6),
            "sharpe": round(self.sharpe, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "annualised_turnover": round(self.annualised_turnover, 6),
            "n_trades": self.n_trades,
            "hhi_concentration": round(self.hhi_concentration, 6),
        }


# ---------------------------------------------------------------------------
# Public scoring entry point
# ---------------------------------------------------------------------------

def score_backtest(
    result: BacktestResult,
    cfg: ScoringConfig | None = None,
) -> ScoreResult:
    """Score a backtest result with strict risk-aware metrics.

    Parameters
    ----------
    result:
        BacktestResult from backtest.run_backtest.
    cfg:
        ScoringConfig. Uses defaults if None.

    Returns
    -------
    ScoreResult
        Contains pass/fail verdict, scalar score, and component metrics.
    """
    if cfg is None:
        cfg = ScoringConfig()

    fail_reasons = []

    # -- Extract metrics --
    returns = result.returns
    equity = result.equity_curve
    trades = result.trades
    positions = result.positions

    if returns.empty or equity.empty:
        fail_reasons.append("backtest produced no output (empty returns)")
        return ScoreResult(
            passed=False, scalar_score=float("nan"), fail_reasons=fail_reasons
        )

    total_return = result.total_return
    ann_return = result.annualised_return
    max_dd = result.max_drawdown  # negative value
    n_trades = result.n_trades
    ann_turnover = result.annualised_turnover

    sortino = _sortino_ratio(returns, risk_free=cfg.sortino_risk_free)
    sharpe = _sharpe_ratio(returns)
    hhi = _hhi_concentration(positions)

    # -- NaN guard --
    if any(math.isnan(v) for v in [total_return, ann_return, sortino, sharpe]):
        fail_reasons.append("NaN detected in core metrics")

    # -- Hard-fail rules --
    if n_trades < cfg.min_trades:
        fail_reasons.append(
            f"too few trades: {n_trades} < {cfg.min_trades} (min_trades)"
        )

    if abs(max_dd) > cfg.max_drawdown_threshold:
        fail_reasons.append(
            f"max drawdown too large: {abs(max_dd):.2%} > {cfg.max_drawdown_threshold:.2%}"
        )

    if ann_turnover > cfg.max_annual_turnover:
        fail_reasons.append(
            f"turnover too high: {ann_turnover:.2f}x > {cfg.max_annual_turnover:.2f}x"
        )

    if hhi > cfg.concentration_hhi_threshold:
        fail_reasons.append(
            f"concentration (HHI={hhi:.3f}) exceeds threshold {cfg.concentration_hhi_threshold}"
        )

    # Check for NaN in returns
    if returns.isna().any():
        fail_reasons.append("NaN values in returns series")

    passed = len(fail_reasons) == 0

    # -- Scalar score (only meaningful when passed) --
    if passed:
        scalar_score = _compute_scalar_score(
            ann_return=ann_return,
            sortino=sortino,
            max_dd=max_dd,
            ann_turnover=ann_turnover,
            cfg=cfg,
        )
    else:
        scalar_score = float("nan")

    return ScoreResult(
        passed=passed,
        scalar_score=scalar_score,
        fail_reasons=fail_reasons,
        total_return=total_return,
        annualised_return=ann_return,
        sortino=sortino,
        sharpe=sharpe,
        max_drawdown=max_dd,
        annualised_turnover=ann_turnover,
        n_trades=n_trades,
        hhi_concentration=hhi,
    )


# ---------------------------------------------------------------------------
# Scalar score formula
# ---------------------------------------------------------------------------

def _compute_scalar_score(
    ann_return: float,
    sortino: float,
    max_dd: float,
    ann_turnover: float,
    cfg: ScoringConfig,
) -> float:
    """Combine metrics into one comparable scalar.

    Formula:
        score = sortino
                - drawdown_penalty_weight * |max_drawdown|
                - turnover_penalty_weight * log1p(annualised_turnover)

    The Sortino ratio is the primary signal. Drawdown and turnover apply
    additive penalties. This penalises risk-taking and churning without
    completely ignoring return.

    Returns NaN if any input is non-finite.
    """
    if not all(math.isfinite(v) for v in [ann_return, sortino, max_dd, ann_turnover]):
        return float("nan")

    score = (
        sortino
        - cfg.drawdown_penalty_weight * abs(max_dd)
        - cfg.turnover_penalty_weight * math.log1p(ann_turnover)
    )
    return float(score)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio."""
    if len(returns) < 20:
        return float("nan")
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    downside_vol = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_vol == 0:
        return float("nan")
    ann_excess = float(excess.mean() * TRADING_DAYS_PER_YEAR)
    return ann_excess / downside_vol


def _sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    if len(returns) < 20:
        return float("nan")
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    vol = float(excess.std(ddof=1))
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def _hhi_concentration(positions: pd.DataFrame) -> float:
    """Herfindahl-Hirschman Index of average symbol-level exposure.

    Returns a value in [0, 1].  1 = entirely concentrated in one symbol.
    0 = perfectly diversified (or no positions).
    """
    if positions.empty:
        return 0.0
    # Average absolute dollar exposure per symbol
    avg_exposure = positions.abs().mean(axis=0)
    total = avg_exposure.sum()
    if total == 0:
        return 0.0
    weights = avg_exposure / total
    return float((weights ** 2).sum())
