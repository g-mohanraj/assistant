"""
download_sample_data.py
-----------------------
Generate synthetic OHLCV sample data and save it to data/raw/.

This script creates CSV files that the rest of the system can load
immediately without any external data source. The generated data uses
geometric Brownian motion — it is NOT real market data.

Usage
~~~~~
    python scripts/download_sample_data.py
    python scripts/download_sample_data.py --symbols AAPL MSFT SPY --years 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_research.data_loader import generate_sample_data
from trading_research.utils import configure_logging, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample OHLCV data."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
        help="Ticker symbols to generate",
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2023-12-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Output directory for CSV files"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    import logging
    logger = logging.getLogger(__name__)

    out_dir = ensure_dir(args.output_dir)

    logger.info(
        "Generating sample data for %s from %s to %s",
        args.symbols, args.start, args.end,
    )

    df = generate_sample_data(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        seed=args.seed,
    )

    # Save one CSV per symbol for clarity
    for sym in args.symbols:
        sym_df = df[df["symbol"] == sym].copy()
        out_path = out_dir / f"{sym}.csv"
        sym_df.to_csv(out_path, index=False)
        logger.info("  Saved %d rows -> %s", len(sym_df), out_path)

    # Also save a combined file
    combined_path = out_dir / "all_symbols.csv"
    df.to_csv(combined_path, index=False)
    logger.info("Saved combined file -> %s (%d rows)", combined_path, len(df))
    print(f"\nSample data written to {out_dir}/")
    print(f"  Symbols : {args.symbols}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    main()
