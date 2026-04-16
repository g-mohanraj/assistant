"""
data_loader.py
--------------
Load and validate OHLCV daily bar data from CSV files.

Expected CSV schema (column names are case-insensitive):
    date, open, high, low, close, volume, symbol

The loader returns a clean, multi-symbol DataFrame with a DatetimeIndex,
sorted deterministically by (date, symbol).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from trading_research.config import Config

logger = logging.getLogger(__name__)

# Canonical column names after normalisation
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume", "symbol"}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a single OHLCV CSV file and return a validated DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: date (DatetimeIndex), open, high, low, close, volume, symbol.
        Sorted by (date, symbol).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    return _validate_and_clean(df, source=str(path))


def load_directory(
    directory: str | Path,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate them.

    Parameters
    ----------
    directory:
        Directory containing one or more OHLCV CSV files.
    symbols:
        If provided, only keep rows for these symbols.

    Returns
    -------
    pd.DataFrame
        Combined, validated, sorted DataFrame.
    """
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    frames = []
    for fp in csv_files:
        try:
            frames.append(load_csv(fp))
        except Exception as exc:
            logger.warning("Skipping %s: %s", fp, exc)

    if not frames:
        raise ValueError(f"No valid CSV files could be loaded from {directory}")

    combined = pd.concat(frames, ignore_index=True)
    combined = _validate_and_clean(combined, source=str(directory))

    if symbols:
        combined = combined[combined["symbol"].isin(symbols)].copy()
        missing = set(symbols) - set(combined["symbol"].unique())
        if missing:
            logger.warning("Symbols not found in data: %s", sorted(missing))

    return combined


def load_from_config(cfg: Config) -> pd.DataFrame:
    """Load data according to a Config object.

    Tries raw_dir first; if empty, tries processed_dir.
    """
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)

    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        df = load_directory(raw_dir, symbols=cfg.data.symbols)
    elif processed_dir.exists() and any(processed_dir.glob("*.csv")):
        df = load_directory(processed_dir, symbols=cfg.data.symbols)
    else:
        logger.info("No data files found; generating synthetic sample data.")
        df = generate_sample_data(
            symbols=cfg.data.symbols,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            seed=cfg.random_seed,
        )

    # Apply date range filter
    df = df[
        (df["date"] >= pd.Timestamp(cfg.data.start_date))
        & (df["date"] <= pd.Timestamp(cfg.data.end_date))
    ].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic sample data (so the repo runs without any external data)
# ---------------------------------------------------------------------------

def generate_sample_data(
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing and quick-starts.

    Uses geometric Brownian motion with realistic parameters.
    Each symbol gets an independent random walk.

    This data is NOT real market data and should not be used for
    real investment decisions.
    """
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range(start=start_date, end=end_date)

    frames = []
    for i, sym in enumerate(symbols):
        n = len(trading_days)
        # GBM parameters
        mu = 0.0003        # daily drift (~7.5% annualised)
        sigma = 0.015      # daily vol (~24% annualised)
        s0 = 100.0 + i * 20.0  # different starting prices

        daily_returns = rng.normal(mu, sigma, n)
        prices = s0 * np.cumprod(1 + daily_returns)

        # Synthesise OHLCV from close prices
        noise = rng.uniform(0.995, 1.005, (n, 4))
        close = prices
        open_ = close * noise[:, 0]
        high = close * np.maximum(noise[:, 1], noise[:, 2]) * 1.003
        low = close * np.minimum(noise[:, 1], noise[:, 2]) * 0.997
        volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        frame = pd.DataFrame(
            {
                "date": trading_days,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": sym,
            }
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_clean(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Normalise column names, validate schema, parse dates, sort."""
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Data source '{source}' is missing required columns: {sorted(missing)}"
        )

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], utc=False)
    if df["date"].isna().any():
        bad_count = df["date"].isna().sum()
        raise ValueError(
            f"Data source '{source}' contains {bad_count} unparseable date values."
        )

    # Cast OHLCV to float
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate no NaN in critical columns
    for col in OHLCV_COLUMNS:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(
                "'%s': column '%s' has %d NaN values; dropping rows.", source, col, nan_count
            )
    df = df.dropna(subset=OHLCV_COLUMNS)

    # Validate OHLC consistency
    bad_ohlc = (df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])
    if bad_ohlc.any():
        logger.warning(
            "'%s': %d rows with inconsistent OHLC values; dropping.", source, bad_ohlc.sum()
        )
        df = df[~bad_ohlc]

    # Ensure symbol is a string
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # Sort deterministically
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Remove timezone info for consistency
    df["date"] = df["date"].dt.tz_localize(None)

    return df


def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a multi-symbol DataFrame to a wide close-price matrix.

    Returns
    -------
    pd.DataFrame
        Index: date, Columns: symbol, Values: close price.
    """
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def pivot_ohlcv(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot a named OHLCV column to wide format (date x symbol)."""
    if column not in OHLCV_COLUMNS:
        raise ValueError(f"column must be one of {OHLCV_COLUMNS}, got '{column}'")
    return df.pivot(index="date", columns="symbol", values=column).sort_index()
