"""Quality threshold filtering for candidate signals."""

import numpy as np

from src.mining.config import MiningConfig
from src.mining.evaluate import EvalResult


def filter_dev_period(
    evaluations: dict[str, EvalResult],
    config: MiningConfig,
) -> dict[str, EvalResult]:
    """Apply development-period quality thresholds.

    Filters on: |ICIR|, hit rate, turnover, decile spread t-stat.
    Uses absolute ICIR to catch signals that predict negatively (flip sign).
    """
    survivors = {}
    for name, ev in evaluations.items():
        # Use absolute ICIR — negative signals are valid (just flip direction)
        abs_icir = abs(ev.dev_icir) if not np.isnan(ev.dev_icir) else 0

        if abs_icir < config.min_icir_dev:
            continue
        if not np.isnan(ev.dev_hit_rate):
            # For negative signals, hit rate is inverted
            effective_hr = ev.dev_hit_rate if ev.dev_icir > 0 else (1 - ev.dev_hit_rate)
            if effective_hr < config.min_hit_rate:
                continue
        if not np.isnan(ev.turnover) and ev.turnover > config.max_turnover:
            continue
        if not np.isnan(ev.dev_spread_t):
            if abs(ev.dev_spread_t) < config.min_spread_t:
                continue

        ev.passed_dev = True
        survivors[name] = ev

    return survivors


def filter_val_period(
    survivors: dict[str, EvalResult],
    config: MiningConfig,
) -> dict[str, EvalResult]:
    """Apply validation-period confirmation threshold."""
    confirmed = {}
    for name, ev in survivors.items():
        abs_val_icir = abs(ev.val_icir) if not np.isnan(ev.val_icir) else 0
        if abs_val_icir >= config.min_icir_val:
            ev.passed_val = True
            confirmed[name] = ev
    return confirmed
