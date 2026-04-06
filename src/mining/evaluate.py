"""Quick evaluation of candidate signals using IC and turnover."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.ic import compute_ic_series, ic_summary
from src.signals.zscore import standardize_signal
from src.signals.report_card import _compute_signal_turnover, _compute_decile_spread


@dataclass
class EvalResult:
    name: str
    dev_ic: float = np.nan
    dev_icir: float = np.nan
    dev_hit_rate: float = np.nan
    dev_t_stat: float = np.nan
    dev_spread: float = np.nan
    dev_spread_t: float = np.nan
    turnover: float = np.nan
    val_ic: float = np.nan
    val_icir: float = np.nan
    n_months_dev: int = 0
    passed_dev: bool = False
    passed_val: bool = False
    passed_dedup: bool = False


def quick_evaluate(
    name: str,
    raw_signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    dev_end: str,
    val_end: str,
) -> EvalResult:
    """Fast evaluation: standardize, compute IC on dev period.

    Does NOT neutralize — raw z-scored IC is used for initial screening.
    Full neutralization is only applied to survivors (expensive).
    """
    result = EvalResult(name=name)

    # Standardize (winsorize + z-score)
    z_signal = standardize_signal(raw_signal, winsorize_pct=0.01)

    # Dev period IC
    ic_dev = compute_ic_series(z_signal, returns, universe, end_date=dev_end)
    if len(ic_dev) < 24:
        return result

    stats = ic_summary(ic_dev)
    result.dev_ic = stats.get("Mean IC", np.nan)
    result.dev_icir = stats.get("ICIR", np.nan)
    result.dev_hit_rate = stats.get("Hit Rate", np.nan)
    result.dev_t_stat = stats.get("t-stat", np.nan)
    result.n_months_dev = stats.get("N Months", 0)

    # Turnover (subsample for speed)
    result.turnover = _compute_signal_turnover(z_signal)

    # Decile spread on dev period
    spread = _compute_decile_spread(z_signal, returns, universe, end_date=dev_end)
    result.dev_spread = spread.get("spread", np.nan)
    result.dev_spread_t = spread.get("spread_t", np.nan)

    return result


def validate_candidate(
    name: str,
    raw_signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    val_start: str,
    val_end: str,
) -> tuple[float, float]:
    """Validation-period IC for survivors."""
    z_signal = standardize_signal(raw_signal, winsorize_pct=0.01)

    # Compute IC only on validation period
    ic_val = compute_ic_series(z_signal, returns, universe, end_date=val_end)
    # Filter to dates after dev period
    val_start_period = pd.Period(val_start, "M")
    ic_val = ic_val[ic_val.index >= val_start_period]

    if len(ic_val) < 12:
        return np.nan, np.nan

    stats = ic_summary(ic_val)
    return stats.get("Mean IC", np.nan), stats.get("ICIR", np.nan)
