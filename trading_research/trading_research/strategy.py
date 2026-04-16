"""
strategy.py
-----------
Strategy interface and baseline strategy implementations.

╔══════════════════════════════════════════════════════════════════╗
║  AGENT-EDITABLE FILE                                             ║
║  This file is the primary surface for coding agent iteration.    ║
║  The backtester and scorer do NOT depend on strategy internals.  ║
║  Safe to modify: add new strategies, tune parameters.            ║
║  Do NOT modify: the StrategyBase interface signature.            ║
╚══════════════════════════════════════════════════════════════════╝

Strategy contract
~~~~~~~~~~~~~~~~~
Each strategy subclass must implement ``generate_signals``.

Input:
    close     — wide DataFrame (date x symbol) of close prices
    features  — dict[str, DataFrame] from features.build_feature_matrix

Output:
    pd.DataFrame with same index (dates) and columns (symbols).
    Values are floats in [0, 1]:
        0   = no position / short signal (long-only: treated as 0)
        > 0 = relative desire to hold (will be normalised by portfolio layer)

Important lookahead rule:
    Signal on date t must use ONLY data available at the END of day t.
    The backtester will execute the resulting position at the OPEN of day t+1.
    Any feature computed on day t is already available by close of day t.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from trading_research.config import StrategyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class StrategyBase(abc.ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement ``generate_signals``. Everything else is fixed.
    """

    name: str = "base"

    @abc.abstractmethod
    def generate_signals(
        self,
        close: pd.DataFrame,
        features: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return a signal DataFrame (date x symbol) with values in [0, 1].

        Parameters
        ----------
        close:
            Wide close-price DataFrame (date x symbol).
        features:
            Dict of feature DataFrames from features.build_feature_matrix.

        Returns
        -------
        pd.DataFrame
            Signal strength per symbol per date. Must have the same index
            and columns as ``close``. Values >= 0.
        """
        ...

    def validate_signals(self, signals: pd.DataFrame, close: pd.DataFrame) -> None:
        """Basic sanity checks on signals (called by experiment runner)."""
        if signals.shape != close.shape:
            raise ValueError(
                f"Signal shape {signals.shape} != close shape {close.shape}"
            )
        if (signals < 0).any().any():
            raise ValueError("Signals contain negative values (long-only system).")
        if signals.isna().all(axis=None):
            raise ValueError("All signals are NaN.")


# ---------------------------------------------------------------------------
# Strategy 1: Momentum
# ---------------------------------------------------------------------------

class MomentumStrategy(StrategyBase):
    """Cross-sectional momentum strategy.

    Each day, rank all symbols by their ``lookback``-day return.
    Assign a long signal to the top ``top_n`` symbols.

    Parameters
    ----------
    lookback:
        Number of days for the return calculation.
    top_n:
        Number of top-ranked symbols to hold. If None, hold all with
        positive momentum.
    """

    name = "momentum"

    def __init__(self, lookback: int = 20, top_n: Optional[int] = 5):
        self.lookback = lookback
        self.top_n = top_n

    def generate_signals(
        self,
        close: pd.DataFrame,
        features: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        ret_key = f"return_{self.lookback}d"
        if ret_key in features:
            ret = features[ret_key]
        else:
            ret = close.pct_change(self.lookback)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for date in close.index:
            row = ret.loc[date]
            valid = row.dropna()
            if valid.empty:
                continue
            if self.top_n is not None:
                top_symbols = valid.nlargest(self.top_n).index
            else:
                top_symbols = valid[valid > 0].index
            signals.loc[date, top_symbols] = 1.0

        return signals


# ---------------------------------------------------------------------------
# Strategy 2: Mean Reversion
# ---------------------------------------------------------------------------

class MeanReversionStrategy(StrategyBase):
    """Time-series mean-reversion strategy based on z-score of returns.

    Long symbols whose z-score of recent returns is below ``z_score_entry``
    (i.e., they have underperformed recently). Exit when z-score rises above
    ``z_score_exit``.

    Parameters
    ----------
    lookback:
        Lookback window for the rolling return and z-score computation.
    z_score_entry:
        Enter long when z-score <= this threshold (negative).
    z_score_exit:
        Exit when z-score >= this threshold.
    top_n:
        Maximum number of symbols to hold simultaneously.
    """

    name = "mean_reversion"

    def __init__(
        self,
        lookback: int = 20,
        z_score_entry: float = -1.5,
        z_score_exit: float = 0.5,
        top_n: Optional[int] = 5,
    ):
        self.lookback = lookback
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.top_n = top_n

    def generate_signals(
        self,
        close: pd.DataFrame,
        features: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        from trading_research.features import z_score, rolling_return

        ret = rolling_return(close, self.lookback)
        zs = z_score(ret, window=self.lookback * 3)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        active: set = set()

        for date in close.index:
            row = zs.loc[date]

            # Exit positions whose z-score has recovered
            exits = {sym for sym in active if not np.isnan(row.get(sym, np.nan)) and row[sym] >= self.z_score_exit}
            active -= exits

            # Enter new positions at oversold levels
            valid = row.dropna()
            candidates = valid[valid <= self.z_score_entry].sort_values()
            for sym in candidates.index:
                if sym not in active:
                    if self.top_n is None or len(active) < self.top_n:
                        active.add(sym)

            signals.loc[date, list(active)] = 1.0

        return signals


# ---------------------------------------------------------------------------
# Strategy 3: Moving Average Crossover
# ---------------------------------------------------------------------------

class MACrossoverStrategy(StrategyBase):
    """Moving average crossover strategy.

    Long when fast MA > slow MA for a symbol. Flat otherwise.

    Parameters
    ----------
    fast_window:
        Short-term moving average period (days).
    slow_window:
        Long-term moving average period (days).
    """

    name = "ma_crossover"

    def __init__(self, fast_window: int = 20, slow_window: int = 50):
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be less than slow_window ({slow_window})"
            )
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(
        self,
        close: pd.DataFrame,
        features: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        from trading_research.features import moving_average

        fast_ma_key = f"ma_{self.fast_window}d"
        slow_ma_key = f"ma_{self.slow_window}d"

        fast_ma_feat = features.get(fast_ma_key)
        fast_ma = fast_ma_feat if fast_ma_feat is not None else moving_average(close, self.fast_window)
        slow_ma_feat = features.get(slow_ma_key)
        slow_ma = slow_ma_feat if slow_ma_feat is not None else moving_average(close, self.slow_window)

        # Signal = 1 where fast > slow, else 0
        raw = (fast_ma > slow_ma).astype(float)
        # Fill NaNs that arise when slow MA hasn't warmed up yet
        raw = raw.fillna(0.0)
        return raw


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ma_crossover": MACrossoverStrategy,
}


def build_strategy(cfg: StrategyConfig) -> StrategyBase:
    """Instantiate a strategy from a StrategyConfig.

    Parameters
    ----------
    cfg:
        StrategyConfig with ``name`` and per-strategy parameter blocks.

    Returns
    -------
    StrategyBase
        Configured strategy instance.

    Raises
    ------
    ValueError
        If ``cfg.name`` is not found in the registry.
    """
    name = cfg.name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )

    cls = STRATEGY_REGISTRY[name]

    if name == "momentum":
        p = cfg.momentum
        return cls(lookback=p.lookback, top_n=p.top_n)
    elif name == "mean_reversion":
        p = cfg.mean_reversion
        return cls(
            lookback=p.lookback,
            z_score_entry=p.z_score_entry,
            z_score_exit=p.z_score_exit,
            top_n=p.top_n,
        )
    elif name == "ma_crossover":
        p = cfg.ma_crossover
        return cls(fast_window=p.fast_window, slow_window=p.slow_window)
    else:
        return cls()
