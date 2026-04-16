# Trading Research System

A local-first autonomous trading research system for backtesting and scoring
simple stock trading strategies.

> **Research only. No live brokerage integration. No real money.**

---

## Project Purpose

This system lets an agent (human or AI) propose simple stock trading
strategies, backtest them against historical data, score them with strict
risk-aware metrics, log experiments in an append-only record, and keep only
improvements. The design favours **robustness, simplicity, and determinism**
over maximising backtest return.

---

## Repo Structure

```
trading_research/
  README.md                      # this file
  requirements.txt               # pip dependencies
  .env.example                   # environment variable template
  pyproject.toml                 # package config
  data/
    raw/                         # place raw CSV files here
    processed/                   # cleaned/processed data (optional)
  configs/
    base.yaml                    # base configuration (edit per experiment)
  trading_research/
    __init__.py
    config.py                    # Config loading and validation (Pydantic)
    data_loader.py               # OHLCV CSV loader + synthetic data generator
    features.py                  # Feature computation (returns, vol, MAs, etc.)
    strategy.py                  # *** AGENT-EDITABLE *** Strategy interface + 3 baselines
    portfolio.py                 # Position sizing and weight normalisation
    backtest.py                  # Daily-bar backtester (next-bar execution)
    scoring.py                   # Risk-aware scoring + hard-fail rules
    experiment_runner.py         # Orchestrates one end-to-end experiment
    experiment_logger.py         # Append-only JSONL experiment log
    validation.py                # Train/val/test splits + walk-forward
    paper_trading_stub.py        # Paper trading placeholder (no live orders)
    utils.py                     # Shared utilities
  scripts/
    download_sample_data.py      # Generate synthetic OHLCV CSV files
    run_baseline.py              # Run all 3 baseline strategies and compare
    run_experiment.py            # Run one experiment from config
    run_walkforward.py           # Walk-forward validation
  experiments/
    experiment_log.jsonl         # Append-only experiment records
    best_configs/                # Saved configs for top experiments
  tests/
    test_data_loader.py
    test_features.py
    test_backtest.py
    test_scoring.py
    test_validation.py
```

---

## Quickstart

### 1. Install dependencies

```bash
cd trading_research
pip install -e ".[dev]"
```

### 2. Generate sample data (optional — system works without it)

```bash
python scripts/download_sample_data.py
```

This creates synthetic OHLCV CSVs in `data/raw/`. The system automatically
generates synthetic data in memory if no CSV files are found.

### 3. Run all baseline strategies

```bash
python scripts/run_baseline.py
```

### 4. Run a single experiment

```bash
python scripts/run_experiment.py --strategy momentum --split validation
```

### 5. Walk-forward validation

```bash
python scripts/run_walkforward.py --strategy ma_crossover
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Data Format

CSV files must contain these columns (case-insensitive):

| Column   | Type    | Description                        |
|----------|---------|------------------------------------|
| date     | string  | ISO date (YYYY-MM-DD)              |
| open     | float   | Opening price                      |
| high     | float   | Daily high                         |
| low      | float   | Daily low                          |
| close    | float   | Closing price                      |
| volume   | float   | Daily traded volume                |
| symbol   | string  | Ticker symbol (e.g. AAPL)          |

Place one or more CSV files in `data/raw/`. The loader accepts a single file
per symbol or a combined file with multiple symbols.

---

## Strategy Interface

Strategies live in `trading_research/strategy.py` and implement `StrategyBase`:

```python
class StrategyBase(abc.ABC):
    @abc.abstractmethod
    def generate_signals(
        self,
        close: pd.DataFrame,      # date x symbol close prices
        features: dict,           # feature name -> date x symbol DataFrame
    ) -> pd.DataFrame:
        """Return signal strengths (date x symbol). Values >= 0."""
        ...
```

**Rules for strategy authors:**
- Signal at date `t` must only use data available by end of day `t`.
- Return values >= 0 (long-only system).
- The backtester shifts signals by 1 day before execution automatically.
- Do **not** modify `portfolio.py`, `backtest.py`, or `scoring.py`.

To add a new strategy:
1. Subclass `StrategyBase` in `strategy.py`.
2. Add it to `STRATEGY_REGISTRY`.
3. Add a config block in `StrategyConfig` if it has parameters.

---

## Experiment Loop

Each experiment follows this pipeline:

```
Config → Data → Features → Strategy → Portfolio → Backtest → Score → Log
```

1. **Config**: Load `configs/base.yaml` (or a custom YAML).
2. **Data**: Load from `data/raw/` or generate synthetic data.
3. **Features**: Compute rolling returns, MAs, volatility, etc.
4. **Strategy**: Generate long signals for each symbol each day.
5. **Portfolio**: Normalise signals into weights (equal weight, capped).
6. **Backtest**: Simulate daily returns with transaction costs.
7. **Score**: Compute Sortino, drawdown, turnover, trade count. Apply hard-fail rules.
8. **Log**: Append one JSON line to `experiments/experiment_log.jsonl`.

### Hard-fail rules (automatic rejection)

| Rule                  | Default threshold |
|-----------------------|-------------------|
| Too few trades        | < 20 trades       |
| Max drawdown too large| > 40%             |
| Turnover too high     | > 20× per year    |
| HHI concentration     | > 0.5             |
| NaN in outputs        | any NaN           |

These are defined in `scoring.py` and are **not** agent-editable.

---

## Train / Validation / Test Separation

The dataset is split chronologically (no shuffle):

```
|←———— 60% train ————→|←— 20% val —→|←— 20% test —→|
oldest                                               newest
```

**Critical rules:**
- The strategy search loop operates on **train + validation only**.
- The **test set is never passed** to `run_experiment()`.
- Call `validation.evaluate_on_test()` only **once**, after the search is complete.
- Multiple test evaluations invalidate the holdout set.

Walk-forward validation generates rolling folds within the train+val window.
This provides a more realistic out-of-sample estimate than a single split.

---

## Scoring Formula

```
score = sortino_ratio
        - 0.5 × |max_drawdown|
        - 0.1 × log(1 + annualised_turnover)
```

A strategy is penalised for large drawdowns and excessive trading. The Sortino
ratio (not raw return) is the primary quality signal, because it accounts for
downside risk specifically.

---

## Why Research-Only

- No broker API credentials, order routing, or live fills.
- No intraday data, options, leverage, or short positions.
- Synthetic data is available for smoke testing.
- All experiments are logged but nothing is committed to real capital.

---

## Safe Files for Agent Mutation

An external coding agent should **only** modify these files:

| File                         | What to change                                    |
|------------------------------|---------------------------------------------------|
| `trading_research/strategy.py` | Add/tune strategies, adjust signal logic        |
| `configs/base.yaml`          | Adjust parameters (lookbacks, windows, thresholds)|
| `trading_research/features.py` | Add new feature computations                  |
| `scripts/`                   | Add new scripts for custom workflows              |

**Do NOT mutate:**

| File                              | Reason                                          |
|-----------------------------------|-------------------------------------------------|
| `trading_research/scoring.py`     | Fixed evaluation — must not be gamed            |
| `trading_research/backtest.py`    | Fixed simulation — lookahead prevention         |
| `trading_research/portfolio.py`   | Fixed constraints — long-only, cap enforcement  |
| `trading_research/validation.py`  | Fixed splits — test set protection              |
| `trading_research/experiment_logger.py` | Append-only integrity                     |

---

## Suggested Next Steps for V2

### 1. Reinforcement Learning
- Add a thin RL environment wrapper over `backtest.py`.
- Use `strategy.py` as the action space (signal weights).
- Reward = scalar score from `scoring.py` (keeps evaluation fixed).
- Suggested library: `gymnasium` with a custom `TradingEnv`.

### 2. Browser / News-Derived Features
- Add `news_features.py` alongside `features.py`.
- Ingest headlines, earnings dates, or sentiment scores as new feature columns.
- Keep the same date x symbol DataFrame convention.
- Gate on a config flag so features can be toggled on/off cleanly.

### 3. Real Paper Trading
- Replace `PaperTrader._emit_order` with a real broker sandbox call.
- Suggested APIs: Alpaca Paper Trading, Interactive Brokers TWS paper account.
- Add reconciliation: compare paper fills with expected weights daily.
- Keep the `paper_trading_stub.py` interface intact.

### 4. Short Selling
- Add a `short_weight` column alongside `long_weight` in the portfolio layer.
- Update `scoring.py` to measure gross vs. net exposure.
- Update `backtest.py` to handle borrow costs.

### 5. Richer Universe
- Extend `download_sample_data.py` to fetch from a free data API (e.g., yfinance).
- Add sector metadata to enable sector-neutral strategies.

---

## Assumptions

1. Data is already split into daily bars (no intraday aggregation needed).
2. Prices are adjusted for splits and dividends (or synthetic data is used).
3. All symbols trade on the same calendar (no cross-market calendar handling).
4. Execution at next-bar open is a reasonable approximation for daily strategies.
5. Transaction costs are symmetric (buy = sell cost).
6. No margin, leverage, or borrowing costs in v1.

## Known Limitations

1. Synthetic data does not model correlations between symbols or market regimes.
2. No survivorship bias correction (real data should use point-in-time universe).
3. Walk-forward folds may have overlapping training windows (by design, for coverage).
4. The scoring formula weights are hand-tuned; they are not validated by theory.
5. No multi-period optimisation; each day's position is chosen independently.
6. Volume ratio feature does not account for market-wide volume changes.
