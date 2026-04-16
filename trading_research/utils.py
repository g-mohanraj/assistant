"""
utils.py
--------
Shared utility functions used across the trading_research package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)
    logger.debug("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def assert_no_lookahead(
    signals: pd.DataFrame,
    feature: pd.DataFrame,
    label: str = "feature",
) -> None:
    """Assert that signals do not peek at future feature values.

    Checks that for every date t, the signal at t was not influenced by
    feature data after t. In practice this verifies that the feature frame
    does not have a smaller index than the signal frame (i.e., future rows
    leaked back).

    This is a lightweight structural check, not a full causal audit.
    Real lookahead protection comes from always using .shift(1) or .pct_change()
    with only past data.

    Parameters
    ----------
    signals:
        Signal DataFrame (date x symbol).
    feature:
        Feature DataFrame (date x symbol) that should only use past data.
    label:
        Name of the feature (for error messages).

    Raises
    ------
    AssertionError
        If the feature index extends further into the future than the signals.
    """
    if signals.empty or feature.empty:
        return

    sig_end = signals.index.max()
    feat_end = feature.index.max()

    if feat_end > sig_end:
        raise AssertionError(
            f"Lookahead detected: {label} has dates up to {feat_end} "
            f"but signals end at {sig_end}."
        )


def align_frames(*dfs: pd.DataFrame) -> tuple:
    """Return copies of DataFrames aligned to their common index and columns."""
    if not dfs:
        return ()
    common_index = dfs[0].index
    common_cols = dfs[0].columns
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)
    return tuple(df.loc[common_index, common_cols] for df in dfs)


def pct_change_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change with explicit NaN for zero denominators."""
    prev = series.shift(periods)
    result = (series - prev) / prev.replace(0, np.nan)
    return result


# ---------------------------------------------------------------------------
# Config / dict helpers
# ---------------------------------------------------------------------------

def dict_hash(d: dict) -> str:
    """Return a short deterministic hash of a dict (for experiment IDs)."""
    serialised = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict."""
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the project root directory (parent of the trading_research package)."""
    return Path(__file__).parent.parent
