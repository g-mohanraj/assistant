"""
paper_trading_stub.py
---------------------
Paper trading stub — shapes the v2 architecture without any real brokerage.

⚠️  THIS MODULE DOES NOT CONNECT TO ANY BROKER. ⚠️
All "orders" are hypothetical and written to a local log file only.
This module exists to define the interface that a real paper-trading
or live-trading layer would implement in a future version.

V2 architecture intent
~~~~~~~~~~~~~~~~~~~~~~~
- PaperTrader reads the latest signal file produced by the experiment runner.
- It computes the hypothetical order needed to move from the current (paper)
  position to the target position.
- It writes the order to a JSON log (paper_orders.jsonl).
- Nothing is sent to any exchange, broker API, or external service.

To graduate to real paper trading (v2+):
1. Replace _emit_order with a call to a brokerage paper-trading sandbox API.
2. Add authentication handling and rate limiting.
3. Add reconciliation logic to compare filled vs. target.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ORDER_LOG = "experiments/paper_orders.jsonl"


class PaperTrader:
    """Hypothetical paper trading executor.

    Parameters
    ----------
    order_log_path:
        Path to the JSONL file where hypothetical orders are written.
    initial_capital:
        Starting paper capital in dollars.
    """

    def __init__(
        self,
        order_log_path: str = DEFAULT_ORDER_LOG,
        initial_capital: float = 100_000.0,
    ):
        self.order_log_path = Path(order_log_path)
        self.order_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self._paper_positions: Dict[str, float] = {}  # symbol -> dollar value
        self._paper_cash: float = initial_capital

    @property
    def paper_portfolio_value(self) -> float:
        return self._paper_cash + sum(self._paper_positions.values())

    def read_latest_signals(self, signals_path: str | Path) -> pd.DataFrame:
        """Read the most recent signal output from a CSV file.

        Expected columns: date, symbol, signal, weight

        Parameters
        ----------
        signals_path:
            Path to a CSV file with columns [date, symbol, signal, weight].

        Returns
        -------
        pd.DataFrame
            Parsed signal DataFrame.
        """
        path = Path(signals_path)
        if not path.exists():
            raise FileNotFoundError(f"Signals file not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "symbol", "weight"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Signals file missing columns: {missing}")
        return df.sort_values(["date", "symbol"])

    def generate_hypothetical_orders(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> list:
        """Compute hypothetical orders to reach target weights from current positions.

        Parameters
        ----------
        target_weights:
            Dict mapping symbol -> target portfolio weight (0..1).
        current_prices:
            Dict mapping symbol -> current market price.

        Returns
        -------
        list[dict]
            List of hypothetical order dicts (no execution).
        """
        total_value = self.paper_portfolio_value
        orders = []

        all_symbols = set(target_weights) | set(self._paper_positions)

        for sym in all_symbols:
            target_dollar = total_value * target_weights.get(sym, 0.0)
            current_dollar = self._paper_positions.get(sym, 0.0)
            delta_dollar = target_dollar - current_dollar

            if abs(delta_dollar) < 1.0:  # ignore tiny orders
                continue

            price = current_prices.get(sym)
            if not price or price <= 0:
                logger.warning("No valid price for %s; skipping order.", sym)
                continue

            shares = delta_dollar / price
            side = "BUY" if shares > 0 else "SELL"

            orders.append(
                {
                    "symbol": sym,
                    "side": side,
                    "shares": round(abs(shares), 4),
                    "price": round(price, 4),
                    "notional": round(abs(delta_dollar), 2),
                    "target_weight": round(target_weights.get(sym, 0.0), 6),
                }
            )

        return orders

    def emit_orders(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        as_of_date: Optional[str] = None,
        notes: str = "",
    ) -> list:
        """Generate and log hypothetical orders.

        This does NOT execute anything. It writes to the paper order log only.

        Parameters
        ----------
        target_weights:
            Target portfolio weights.
        current_prices:
            Current market prices (used for notional calculation).
        as_of_date:
            ISO date string for the order date; defaults to UTC now.
        notes:
            Free-text annotation.

        Returns
        -------
        list[dict]
            The hypothetical orders that were logged.
        """
        orders = self.generate_hypothetical_orders(target_weights, current_prices)

        if not orders:
            logger.info("No hypothetical orders to emit.")
            return []

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "as_of_date": as_of_date or datetime.now(timezone.utc).date().isoformat(),
            "paper_portfolio_value": round(self.paper_portfolio_value, 2),
            "orders": orders,
            "notes": notes,
            "WARNING": "PAPER TRADING ONLY — NO REAL ORDERS SENT",
        }

        with open(self.order_log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        logger.info(
            "Emitted %d hypothetical paper orders (total notional=%.2f)",
            len(orders),
            sum(o["notional"] for o in orders),
        )
        return orders

    def apply_hypothetical_fills(
        self,
        orders: list,
        current_prices: Dict[str, float],
    ) -> None:
        """Update paper positions assuming all orders fill at current_prices.

        In v2 with a real paper broker, this would be replaced by reading
        actual fill reports from the broker API.
        """
        for order in orders:
            sym = order["symbol"]
            shares = order["shares"]
            price = current_prices.get(sym, order["price"])
            value = shares * price

            if order["side"] == "BUY":
                self._paper_positions[sym] = self._paper_positions.get(sym, 0.0) + value
                self._paper_cash -= value
            else:
                self._paper_positions[sym] = self._paper_positions.get(sym, 0.0) - value
                self._paper_cash += value

        # Clean up zero positions
        self._paper_positions = {
            sym: v for sym, v in self._paper_positions.items() if abs(v) > 0.01
        }
