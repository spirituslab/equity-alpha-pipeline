"""Signal combination methods.

Combines K neutralized signals into a single composite alpha score per stock.
"""

import numpy as np
import pandas as pd

from src.analytics.ic import compute_ic_series


def equal_weight_combine(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight average of neutralized z-scores.

    alpha_{i,t} = (1/K) * sum_k(z_k_{i,t})

    Simple, robust baseline. No estimation error from weight selection.
    """
    aligned = _align_signals(signals)
    stacked = np.stack([df.values for df in aligned.values()], axis=0)
    # Mean ignoring NaN
    with np.errstate(all="ignore"):
        composite = np.nanmean(stacked, axis=0)
    return pd.DataFrame(composite, index=list(aligned.values())[0].index,
                        columns=list(aligned.values())[0].columns)


def ic_weighted_combine(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    lookback: int = 36,
    min_ic_months: int = 12,
) -> pd.DataFrame:
    """IC-weighted signal combination.

    alpha_{i,t} = sum_k(w_{k,t} * z_k_{i,t})

    where w_{k,t} = trailing IC_k (rolling lookback-month average rank IC)
    normalized to sum to 1, with floor at 0 (drop negative-IC signals).

    Adapts to signal strength over time — gives more weight to signals
    with stronger recent predictive power.
    """
    aligned = _align_signals(signals)
    dates = list(aligned.values())[0].index
    columns = list(aligned.values())[0].columns

    # Compute IC series for each signal
    ic_series = {}
    for name, sig in aligned.items():
        ic_series[name] = compute_ic_series(sig, returns, universe)

    composite = pd.DataFrame(0.0, index=dates, columns=columns)

    for t_idx, t in enumerate(dates):
        # Get trailing IC for each signal
        weights = {}
        for name, ic in ic_series.items():
            # Use IC values up to (but not including) current date
            trailing = ic[ic.index < t]
            if len(trailing) < min_ic_months:
                weights[name] = 0.0
                continue
            # Rolling mean of last `lookback` IC values
            recent = trailing.iloc[-lookback:] if len(trailing) > lookback else trailing
            mean_ic = recent.mean()
            weights[name] = max(mean_ic, 0.0)  # floor at 0

        # Normalize weights
        total = sum(weights.values())
        if total <= 0:
            # Fallback to equal weight if all ICs are negative
            total = len(weights)
            weights = {k: 1.0 / total for k in weights}
        else:
            weights = {k: v / total for k, v in weights.items()}

        # Compute weighted combination
        for name, w in weights.items():
            if w > 0 and t in aligned[name].index:
                vals = aligned[name].loc[t]
                composite.loc[t] += w * vals.fillna(0)

    return composite


def inverse_vol_combine(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    lookback: int = 36,
) -> pd.DataFrame:
    """Inverse-volatility-weighted signal combination.

    Weight each signal by 1 / std(IC) — more weight to signals with
    more consistent (less volatile) predictive power.
    """
    aligned = _align_signals(signals)
    dates = list(aligned.values())[0].index
    columns = list(aligned.values())[0].columns

    ic_series = {}
    for name, sig in aligned.items():
        ic_series[name] = compute_ic_series(sig, returns, universe)

    composite = pd.DataFrame(0.0, index=dates, columns=columns)

    for t_idx, t in enumerate(dates):
        weights = {}
        for name, ic in ic_series.items():
            trailing = ic[ic.index < t]
            if len(trailing) < 12:
                weights[name] = 1.0
                continue
            recent = trailing.iloc[-lookback:] if len(trailing) > lookback else trailing
            ic_std = recent.std()
            weights[name] = 1.0 / ic_std if ic_std > 0.01 else 1.0

        total = sum(weights.values())
        if total <= 0:
            total = len(weights)
            weights = {k: 1.0 / total for k in weights}
        else:
            weights = {k: v / total for k, v in weights.items()}

        for name, w in weights.items():
            if w > 0 and t in aligned[name].index:
                composite.loc[t] += w * aligned[name].loc[t].fillna(0)

    return composite


def combine_signals(
    signals: dict[str, pd.DataFrame],
    method: str = "equal",
    returns: pd.DataFrame = None,
    universe: pd.DataFrame = None,
    lookback: int = 36,
    train_window: int = 60,
    purge_gap: int = 2,
) -> pd.DataFrame:
    """Dispatch to the appropriate combination method.

    Args:
        signals: dict mapping signal name -> (date x gvkey) neutralized z-scores
        method: "equal", "ic_weighted", "inverse_vol", "ridge", "elastic_net", or "xgboost"
        returns: required for ic_weighted, inverse_vol, and ML methods
        universe: required for ic_weighted, inverse_vol, and ML methods
        lookback: IC lookback window for ic_weighted / inverse_vol
        train_window: training window for ML methods
        purge_gap: purge gap for ML methods
    """
    if method == "equal":
        return equal_weight_combine(signals)
    elif method == "ic_weighted":
        if returns is None or universe is None:
            raise ValueError("ic_weighted requires returns and universe DataFrames")
        return ic_weighted_combine(signals, returns, universe, lookback=lookback)
    elif method == "inverse_vol":
        if returns is None or universe is None:
            raise ValueError("inverse_vol requires returns and universe DataFrames")
        return inverse_vol_combine(signals, returns, universe, lookback=lookback)
    elif method in ("ridge", "elastic_net", "xgboost"):
        if returns is None or universe is None:
            raise ValueError(f"{method} requires returns and universe DataFrames")
        from src.ml.models import ml_combine
        return ml_combine(
            signals, returns, universe,
            model_type=method,
            train_window=train_window,
            purge_gap=purge_gap,
        )
    else:
        raise ValueError(f"Unknown combination method: {method}. "
                         f"Use 'equal', 'ic_weighted', 'ridge', 'elastic_net', or 'xgboost'.")


def _align_signals(signals: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align all signals to common dates and stocks."""
    if not signals:
        raise ValueError("No signals provided")

    # Find common dates and stocks
    common_dates = None
    common_stocks = None
    for sig in signals.values():
        if common_dates is None:
            common_dates = sig.index
            common_stocks = sig.columns
        else:
            common_dates = common_dates.intersection(sig.index)
            common_stocks = common_stocks.intersection(sig.columns)

    return {name: sig.loc[common_dates, common_stocks] for name, sig in signals.items()}
