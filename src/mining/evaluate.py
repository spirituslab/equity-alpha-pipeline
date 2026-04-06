"""Quick evaluation of candidate signals using IC and turnover."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.ic import compute_ic_series, ic_summary
from src.signals.zscore import standardize_signal
from src.signals.report_card import _compute_signal_turnover, _compute_decile_spread
from src.gpu.backend import GPU_AVAILABLE


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


def batch_evaluate_gpu(
    raw_signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    dev_end: str,
) -> dict[str, EvalResult]:
    """GPU-batched evaluation: compute IC for all signals at once.

    Much faster than calling quick_evaluate() in a loop.
    """
    from src.gpu.ic_batch import batch_compute_ic

    # Standardize all signals first (CPU — fast enough)
    z_signals = {}
    for name, raw in raw_signals.items():
        z_signals[name] = standardize_signal(raw, winsorize_pct=0.01)

    # Batch IC computation on GPU
    print(f"    GPU batch IC for {len(z_signals)} signals...")
    ic_dict = batch_compute_ic(z_signals, returns, universe, end_date=dev_end)

    # Build EvalResults
    results = {}
    for name in raw_signals:
        ev = EvalResult(name=name)
        if name in ic_dict:
            ic = ic_dict[name]
            stats = ic_summary(ic)
            ev.dev_ic = stats.get("Mean IC", np.nan)
            ev.dev_icir = stats.get("ICIR", np.nan)
            ev.dev_hit_rate = stats.get("Hit Rate", np.nan)
            ev.dev_t_stat = stats.get("t-stat", np.nan)
            ev.n_months_dev = stats.get("N Months", 0)

        # Turnover (CPU — per-signal, fast)
        if name in z_signals:
            ev.turnover = _compute_signal_turnover(z_signals[name])

        # Decile spread (CPU — per-signal)
        if name in z_signals:
            spread = _compute_decile_spread(z_signals[name], returns, universe, end_date=dev_end)
            ev.dev_spread = spread.get("spread", np.nan)
            ev.dev_spread_t = spread.get("spread_t", np.nan)

        results[name] = ev

    return results
