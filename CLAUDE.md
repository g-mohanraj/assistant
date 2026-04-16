# CLAUDE.md — Agent Instructions for Trading Research System

This file is read automatically by Claude Code at session start.
It tells any coding agent exactly what it is allowed to change, what it must not touch,
and how to run the experiment loop safely.

---

## What this repo is

A local-first trading research system for backtesting and scoring simple
long-only daily-bar stock strategies. It is **research-only** — no live trading,
no broker connection, no real money.

---

## Ground rules for agents

1. **Do not modify the evaluator.** The files `trading_research/scoring.py`,
   `trading_research/backtest.py`, `trading_research/portfolio.py`, and
   `trading_research/validation.py` are fixed. Changing them to make a strategy
   score higher is cheating and invalidates all results.

2. **Do not leak the test set.** Never pass `split.test` to `run_experiment()`.
   The test set is reserved for one final evaluation only. Use `split.train` and
   `split.validation` during the search loop.

3. **Always run tests before committing.** Run `pytest tests/ -q` and confirm
   54 tests pass. If you break a test, fix it before committing.

4. **Append to the experiment log, never delete it.** The file
   `experiments/experiment_log.jsonl` is append-only. Do not truncate or rewrite it.

5. **Commit after every meaningful change.** One experiment iteration = one commit.

---

## Safe files to edit

| File | What to do here |
|---|---|
| `trading_research/strategy.py` | Add strategies, tune signal logic — **primary edit surface** |
| `trading_research/features.py` | Add new feature computations |
| `configs/base.yaml` | Adjust parameters (lookbacks, thresholds, symbols, cost assumptions) |
| `scripts/` | Add new workflow scripts |
| `tests/` | Add tests for new strategies or features |

## Files you must NOT edit

| File | Reason |
|---|---|
| `trading_research/scoring.py` | Fixed evaluator — must not be gamed |
| `trading_research/backtest.py` | Fixed simulator — lookahead prevention |
| `trading_research/portfolio.py` | Fixed constraints — long-only, cap enforcement |
| `trading_research/validation.py` | Fixed splits — test set protection |
| `trading_research/experiment_logger.py` | Append-only log integrity |
| `trading_research/experiment_runner.py` | Orchestration — safe to read, avoid editing |

---

## How to add a new strategy (step by step)

1. Open `trading_research/strategy.py`.
2. Subclass `StrategyBase` and implement `generate_signals(close, features)`.
   - `close`: wide DataFrame, shape (dates × symbols), close prices.
   - `features`: dict of named DataFrames from `features.build_feature_matrix`.
   - Return a DataFrame of the same shape with values >= 0.
   - **Signal at date t must only use data available at close of day t.**
     The backtester shifts by 1 day automatically — do not shift manually.
3. Add the class to `STRATEGY_REGISTRY` at the bottom of `strategy.py`.
4. Add a config block to `StrategyConfig` in `trading_research/config.py` if the
   strategy has parameters.
5. Add the same block to `configs/base.yaml`.
6. Run `pytest tests/ -q` to confirm nothing is broken.
7. Run `python scripts/run_experiment.py --strategy your_strategy_name --notes "rationale"`.
8. Check `experiments/experiment_log.jsonl` for the result.

---

## How to add a new feature

1. Open `trading_research/features.py`.
2. Write a pure function: takes wide DataFrames, returns a wide DataFrame of
   the same shape. No side effects, no randomness.
3. Add it to `build_feature_matrix` so strategies can access it by name.
4. Reference it in `generate_signals` via `features["your_feature_name"]`.

---

## Experiment loop (the safe iteration pattern)

```
Edit strategy.py or configs/base.yaml
  → python scripts/run_experiment.py --strategy NAME --notes "what changed and why"
  → check scalar_score and pass/fail in the log
  → if improved: keep the change
  → if not improved: revert the change
  → repeat
```

After settling on a strategy, run walk-forward to confirm robustness:
```
python scripts/run_walkforward.py --strategy NAME
```

Only run the final holdout test **once**:
```python
from trading_research.validation import evaluate_on_test
```

---

## Hard-fail rules (automatic rejection — do not try to work around these)

| Rule | Default |
|---|---|
| Fewer than 20 trades | rejected |
| Max drawdown > 40% | rejected |
| Annualised turnover > 20× | rejected |
| HHI concentration > 0.5 | rejected |
| NaN in any output | rejected |

These are defined in `trading_research/scoring.py` (`ScoringConfig`) and also
configurable in `configs/base.yaml` under the `scoring:` block — but tightening
them is encouraged; loosening them defeats the purpose.

---

## Scoring formula (higher = better)

```
score = sortino_ratio
        - 0.5 × |max_drawdown|
        - 0.1 × log(1 + annualised_turnover)
```

Optimise this score. Do not optimise raw return alone.

---

## Running the test suite

```bash
pytest tests/ -q          # quick pass/fail
pytest tests/ -v          # verbose with test names
pytest tests/ --tb=short  # short tracebacks on failure
```

Expected: **54 tests pass, 0 fail.**

Tests cover:
- No lookahead in features and signal execution (`test_features.py`, `test_backtest.py`)
- Backtest determinism and cost correctness (`test_backtest.py`)
- Scoring hard-fail rules (`test_scoring.py`)
- Split non-overlap and chronological ordering (`test_validation.py`)
- Data loader schema validation (`test_data_loader.py`)

---

## Data format

CSV files in `data/raw/` must have these columns (case-insensitive):

```
date, open, high, low, close, volume, symbol
```

If `data/raw/` is empty, synthetic GBM data is generated automatically using
`random_seed: 42` from `configs/base.yaml`. This is enough for smoke testing
but not for meaningful research.

---

## Key architecture facts

- **Execution model**: signal on day `t` → fills at open of day `t+1`. This is
  enforced by a `.shift(1)` in `backtest.py:run_backtest`. Never shift signals
  yourself inside a strategy.
- **Long-only**: negative signal values are clipped to 0 in `portfolio.py`.
- **Equal weight**: position sizing is equal weight across active signals,
  subject to `max_positions` and `max_position_size` caps in config.
- **Transaction costs**: `transaction_cost_bps + slippage_bps` applied one-way
  on each trade's notional value.
- **Reproducibility**: set `random_seed` in `configs/base.yaml` or `.env`.
  All synthetic data generation is seeded.

---

## Commit message convention

```
<verb> <what changed>: <one-line rationale>

Extended notes if needed.
```

Examples:
- `Add dual-momentum strategy: cross-sectional + time-series filter`
- `Tune momentum lookback from 20d to 40d: reduces turnover on validation`
- `Add RSI feature: needed for mean-reversion signal refinement`

---

## What NOT to build in v1

- No short selling
- No leverage
- No intraday data
- No options
- No live broker calls
- No LLM calls inside strategy logic
- No browser/news ingestion (feature stubs are fine)
