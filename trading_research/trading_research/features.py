"""
features.py
-----------
Compute features from OHLCV data.

All feature functions operate on wide DataFrames (date x symbol) and return
DataFrames of the same shape. The helper ``build_feature_matrix`` assembles
a dict of named feature frames from a raw OHLCV DataFrame.

Design rules
~~~~~~~~~~~~
- No forward-looking computation: every feature at date t uses only data
  available strictly before or at t.
- All functions are pure: same inputs -> same outputs.
- New features can be added here without touching strategy.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from trading_research.config import Config, FeaturesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual feature functions (wide DataFrames: date x symbol)
# ---------------------------------------------------------------------------

def rolling_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Percentage return over the last ``window`` trading days.

    return_t = close_t / close_{t-window} - 1
    """
    return close.pct_change(periods=window)


def rolling_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation of daily log returns.

    Uses a minimum of ``window // 2`` observations to avoid excessive NaNs
    near the start of the series.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window // 2).std()


def moving_average(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average of close prices."""
    return close.rolling(window=window, min_periods=window // 2).mean()


def ma_ratio(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Close price divided by its moving average (price-to-MA ratio)."""
    ma = moving_average(close, window)
    return close / ma - 1.0


def volume_ratio(
    volume: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Today's volume divided by its rolling mean.

    Values above 1 indicate above-average volume.
    """
    vol_ma = volume.rolling(window=window, min_periods=window // 2).mean()
    return volume / vol_ma


def momentum_score(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Raw momentum: return over ``window`` days (alias for rolling_return)."""
    return rolling_return(close, window)


def z_score(series: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score of values in each column.

    z_t = (x_t - mean_{t-window:t}) / std_{t-window:t}
    """
    mu = series.rolling(window=window, min_periods=window // 2).mean()
    sigma = series.rolling(window=window, min_periods=window // 2).std()
    return (series - mu) / sigma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: FeaturesConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """Compute all baseline features and return as a named dictionary.

    Parameters
    ----------
    close:
        Wide DataFrame: date x symbol close prices.
    volume:
        Wide DataFrame: date x symbol volumes.
    cfg:
        FeaturesConfig. Defaults to FeaturesConfig() if None.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are descriptive feature names; values are date x symbol DataFrames.
    """
    if cfg is None:
        cfg = FeaturesConfig()

    features: Dict[str, pd.DataFrame] = {}

    # Rolling returns at each configured window
    for w in cfg.return_windows:
        features[f"return_{w}d"] = rolling_return(close, w)

    # Rolling volatility
    features[f"volatility_{cfg.volatility_window}d"] = rolling_volatility(
        close, cfg.volatility_window
    )

    # Moving averages and their ratios to price
    for w in cfg.ma_windows:
        features[f"ma_{w}d"] = moving_average(close, w)
        features[f"ma_ratio_{w}d"] = ma_ratio(close, w)

    # Volume ratio
    features[f"volume_ratio_{cfg.volume_ma_window}d"] = volume_ratio(
        volume, cfg.volume_ma_window
    )

    # Momentum z-score (20d default)
    features["momentum_z20"] = z_score(rolling_return(close, 20), 60)

    logger.debug("Built %d feature frames.", len(features))
    return features


def build_feature_matrix_from_df(
    df: pd.DataFrame,
    cfg: FeaturesConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper that accepts a long-format OHLCV DataFrame.

    Parameters
    ----------
    df:
        Long DataFrame with columns: date, open, high, low, close, volume, symbol.
    cfg:
        FeaturesConfig instance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same as ``build_feature_matrix``.
    """
    from trading_research.data_loader import pivot_ohlcv

    close = pivot_ohlcv(df, "close")
    volume = pivot_ohlcv(df, "volume")
    return build_feature_matrix(close, volume, cfg)
