"""
config.py
---------
Load and validate configuration from YAML files and environment variables.

The Config object is the single source of truth passed through the pipeline.
All modules receive a Config instance rather than reading files themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"


class SplitsConfig(BaseModel):
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    walkforward_train_years: int = 2
    walkforward_val_years: int = 1
    walkforward_step_months: int = 6

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SplitsConfig":
        total = round(self.train + self.validation + self.test, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"splits.train + splits.validation + splits.test must equal 1.0, got {total}"
            )
        return self


class FeaturesConfig(BaseModel):
    return_windows: List[int] = Field(default_factory=lambda: [5, 20, 50])
    volatility_window: int = 20
    ma_windows: List[int] = Field(default_factory=lambda: [20, 50, 200])
    volume_ma_window: int = 20


class MomentumStrategyConfig(BaseModel):
    lookback: int = 20
    top_n: int = 5


class MeanReversionStrategyConfig(BaseModel):
    lookback: int = 20
    z_score_entry: float = -1.5
    z_score_exit: float = 0.5
    top_n: int = 5


class MACrossoverStrategyConfig(BaseModel):
    fast_window: int = 20
    slow_window: int = 50


class StrategyConfig(BaseModel):
    name: str = "momentum"
    momentum: MomentumStrategyConfig = Field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionStrategyConfig = Field(
        default_factory=MeanReversionStrategyConfig
    )
    ma_crossover: MACrossoverStrategyConfig = Field(
        default_factory=MACrossoverStrategyConfig
    )


class PortfolioConfig(BaseModel):
    max_positions: Optional[int] = 10
    max_position_size: Optional[float] = 0.20
    cash_when_no_signal: bool = True


class BacktestConfig(BaseModel):
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 100_000.0


class ScoringConfig(BaseModel):
    min_trades: int = 20
    max_drawdown_threshold: float = 0.40
    max_annual_turnover: float = 20.0
    concentration_hhi_threshold: float = 0.5
    sortino_risk_free: float = 0.0
    drawdown_penalty_weight: float = 0.5
    turnover_penalty_weight: float = 0.1


class ExperimentConfig(BaseModel):
    log_path: str = "experiments/experiment_log.jsonl"
    best_configs_dir: str = "experiments/best_configs"
    notes: str = ""


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "configs/base.yaml") -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML config file. Missing keys fall back to model defaults.

    Returns
    -------
    Config
        A fully validated Config object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    # Allow environment variable overrides for a small set of top-level keys
    if os.getenv("DATA_RAW_DIR"):
        raw.setdefault("data", {})["raw_dir"] = os.environ["DATA_RAW_DIR"]
    if os.getenv("DATA_PROCESSED_DIR"):
        raw.setdefault("data", {})["processed_dir"] = os.environ["DATA_PROCESSED_DIR"]
    if os.getenv("EXPERIMENT_LOG_PATH"):
        raw.setdefault("experiment", {})["log_path"] = os.environ["EXPERIMENT_LOG_PATH"]
    if os.getenv("RANDOM_SEED"):
        raw["random_seed"] = int(os.environ["RANDOM_SEED"])

    return Config.model_validate(raw)


def default_config() -> Config:
    """Return a default Config without reading any file."""
    return Config()
