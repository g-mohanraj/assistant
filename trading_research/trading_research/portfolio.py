"""
portfolio.py
------------
Convert raw strategy signals into dollar target weights.

Rules (v1 — long-only)
~~~~~~~~~~~~~~~~~~~~~~~
1. Accept raw signal strengths (>= 0) from the strategy layer.
2. Zero out any negative values (long-only guard).
3. Apply max_positions cap: keep the top-N signals if more are active.
4. Apply per-position size cap (max_position_size).
5. Normalise remaining signals so weights sum to 1.0 (or 0.0 if no signals).
6. Return a weights DataFrame (date x symbol).

This module is intentionally NOT agent-editable. It enforces portfolio
constraints that the evaluator relies on.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading_research.config import PortfolioConfig

logger = logging.getLogger(__name__)


def compute_weights(
    signals: pd.DataFrame,
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """Convert signal strengths into normalised portfolio weights.

    Parameters
    ----------
    signals:
        Wide DataFrame (date x symbol) with raw signal values >= 0.
    cfg:
        PortfolioConfig. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        Weights DataFrame (date x symbol). Each row sums to <= 1.0.
        A row sum < 1.0 implies the remainder is held in cash.
    """
    if cfg is None:
        cfg = PortfolioConfig()

    weights = signals.copy().astype(float)

    # Long-only guard
    weights = weights.clip(lower=0.0)

    n_dates = len(weights)
    for i in range(n_dates):
        row = weights.iloc[i]
        active = row[row > 0]

        if active.empty:
            weights.iloc[i] = 0.0
            continue

        # Apply max_positions cap (keep top-N by signal strength)
        if cfg.max_positions is not None and len(active) > cfg.max_positions:
            top_syms = active.nlargest(cfg.max_positions).index
            # Zero out symbols not in top-N
            mask = pd.Series(False, index=row.index)
            mask[top_syms] = True
            weights.iloc[i, ~mask.values] = 0.0
            active = weights.iloc[i][weights.iloc[i] > 0]

        # Equal weight (ignore signal magnitude for now — simpler and less overfit-prone)
        n_active = (active > 0).sum()
        equal_weight = 1.0 / n_active

        # Apply per-position size cap
        if cfg.max_position_size is not None:
            per_weight = min(equal_weight, cfg.max_position_size)
        else:
            per_weight = equal_weight

        # Set weights
        for sym in weights.columns:
            if weights.iloc[i][sym] > 0:
                weights.iloc[i][weights.columns.get_loc(sym)] = per_weight
            else:
                weights.iloc[i][weights.columns.get_loc(sym)] = 0.0

    return weights


def compute_weights_vectorised(
    signals: pd.DataFrame,
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """Faster vectorised weight computation (no stateful logic).

    Uses equal weighting across active positions with max-positions and
    max-size caps applied per row. Equivalent to compute_weights for
    strategies that don't rely on carry-forward state.

    Parameters
    ----------
    signals:
        Wide DataFrame (date x symbol). Values > 0 indicate an active position.
    cfg:
        PortfolioConfig.

    Returns
    -------
    pd.DataFrame
        Normalised weights (date x symbol).
    """
    if cfg is None:
        cfg = PortfolioConfig()

    signals = signals.clip(lower=0.0)
    active = (signals > 0).astype(float)

    # Apply max_positions: for each row keep only top-N columns by signal value
    if cfg.max_positions is not None:
        # Rank signals per row; keep top-N
        ranks = signals.rank(axis=1, ascending=False, method="first")
        cap_mask = ranks <= cfg.max_positions
        active = active * cap_mask.astype(float)

    # Equal weight
    n_active = active.sum(axis=1).replace(0, np.nan)
    equal_w = active.div(n_active, axis=0).fillna(0.0)

    # Apply per-position cap
    if cfg.max_position_size is not None:
        equal_w = equal_w.clip(upper=cfg.max_position_size)

    return equal_w
