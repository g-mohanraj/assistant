"""
experiment_logger.py
--------------------
Append-only experiment log.

Each experiment is a single JSON line written to a .jsonl file.
The log is append-only: we never overwrite or delete past experiments.
This lets you audit the full search history and reproduce any result.

Schema of each record
~~~~~~~~~~~~~~~~~~~~~
    {
        "timestamp"     : ISO-8601 UTC timestamp,
        "experiment_id" : sequential integer,
        "strategy_name" : str,
        "config"        : dict (full config snapshot),
        "data_window"   : {"start": ..., "end": ..., "split": "validation"},
        "metrics"       : dict (component metrics),
        "scalar_score"  : float or null,
        "passed"        : bool,
        "fail_reasons"  : list[str],
        "git_hash"      : str or null,
        "notes"         : str
    }
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_git_hash() -> Optional[str]:
    """Return the current HEAD commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_existing_log(path: Path) -> list:
    """Read existing JSONL records (for counting experiment IDs)."""
    records = []
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed log line in %s", path)
    return records


def log_experiment(
    log_path: str | Path,
    strategy_name: str,
    config: Dict[str, Any],
    data_window: Dict[str, str],
    metrics: Dict[str, Any],
    scalar_score: Optional[float],
    passed: bool,
    fail_reasons: list,
    notes: str = "",
) -> Dict[str, Any]:
    """Append one experiment record to the JSONL log.

    Parameters
    ----------
    log_path:
        Path to the .jsonl log file. Created if absent (parent dirs too).
    strategy_name:
        Name of the strategy evaluated.
    config:
        Full config dict snapshot (serialisable).
    data_window:
        Dict with keys ``start``, ``end``, and ``split`` (e.g. "validation").
    metrics:
        Dict of component metrics from ScoreResult.to_dict().
    scalar_score:
        The scalar score (or None / NaN on failure).
    passed:
        Whether the experiment passed all hard-fail checks.
    fail_reasons:
        List of failure reason strings.
    notes:
        Free-text field for agent rationale.

    Returns
    -------
    dict
        The record that was written.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_log(log_path)
    experiment_id = len(existing)

    # Normalise scalar_score to JSON-compatible value
    if scalar_score is not None and (
        scalar_score != scalar_score  # NaN check
    ):
        scalar_score = None

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "strategy_name": strategy_name,
        "config": _make_serialisable(config),
        "data_window": data_window,
        "metrics": _make_serialisable(metrics),
        "scalar_score": scalar_score,
        "passed": passed,
        "fail_reasons": fail_reasons,
        "git_hash": _get_git_hash(),
        "notes": notes,
    }

    with open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.info(
        "Logged experiment #%d | strategy=%s | passed=%s | score=%s",
        experiment_id,
        strategy_name,
        passed,
        f"{scalar_score:.4f}" if scalar_score is not None else "N/A",
    )
    return record


def load_log(log_path: str | Path) -> list:
    """Load all experiment records from the JSONL log.

    Returns
    -------
    list[dict]
        All records in chronological order.
    """
    return _load_existing_log(Path(log_path))


def best_experiments(
    log_path: str | Path,
    n: int = 10,
    strategy_name: Optional[str] = None,
) -> list:
    """Return the top-N experiments by scalar_score.

    Parameters
    ----------
    log_path:
        Path to the JSONL log.
    n:
        Number of top experiments to return.
    strategy_name:
        If set, filter to only this strategy.

    Returns
    -------
    list[dict]
        Sorted by scalar_score descending.
    """
    records = load_log(log_path)
    passed = [r for r in records if r.get("passed") and r.get("scalar_score") is not None]
    if strategy_name:
        passed = [r for r in passed if r.get("strategy_name") == strategy_name]
    return sorted(passed, key=lambda r: r["scalar_score"], reverse=True)[:n]


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    return obj
