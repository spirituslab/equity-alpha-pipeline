"""Information Coefficient computation for signal evaluation."""

import numpy as np
import pandas as pd


def compute_ic_series(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    end_date: str = None,
    start_date: str = None,
    purge_start: str = None,
    purge_end: str = None,
) -> pd.Series:
    """Compute monthly rank IC (Spearman) for a signal.

    IC_t = Spearman correlation between signal at t and returns at t+1,
    computed cross-sectionally across universe members.

    Args:
        signal: (date x gvkey) signal values
        returns: (date x gvkey) stock returns (decimal)
        universe: (date x gvkey) boolean mask for investable universe
        end_date: if provided, only use dates up to this period
        start_date: if provided, only use dates from this period onward
        purge_start: exclude dates in [purge_start, purge_end] (fold boundary purge)
        purge_end: exclude dates in [purge_start, purge_end]
    """
    ic_values = {}

    dates = signal.index
    if end_date is not None:
        dates = dates[dates <= pd.Period(end_date, "M")]
    if start_date is not None:
        dates = dates[dates >= pd.Period(start_date, "M")]
    if purge_start is not None and purge_end is not None:
        purge_s = pd.Period(purge_start, "M")
        purge_e = pd.Period(purge_end, "M")
        dates = dates[(dates < purge_s) | (dates > purge_e)]

    for t in dates:
        t_plus_1 = t + 1
        if t_plus_1 not in returns.index:
            continue

        # Filter to universe
        if t not in universe.index:
            continue
        members = universe.loc[t]
        members = members[members].index  # stocks where universe == True

        s_vals = signal.loc[t].reindex(members).dropna()
        r_vals = returns.loc[t_plus_1].reindex(members).dropna()

        common = np.intersect1d(s_vals.index.values, r_vals.index.values)
        if len(common) < 20:
            continue

        s = s_vals.reindex(common).replace([np.inf, -np.inf], np.nan).dropna()
        r = r_vals.reindex(s.index)

        valid = s.notna() & r.notna()
        s = s[valid]
        r = r[valid]

        if len(s) < 20:
            continue

        # Manual Spearman: rank then Pearson
        s_ranks = s.values.argsort().argsort().astype(float)
        r_ranks = r.values.argsort().argsort().astype(float)
        corr = np.corrcoef(s_ranks, r_ranks)[0, 1]
        ic_values[t_plus_1] = corr

    return pd.Series(ic_values, dtype=float)


def ic_summary(ic_series: pd.Series) -> dict:
    """IC summary statistics."""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {}
    return {
        "Mean IC": ic.mean(),
        "Std IC": ic.std(),
        "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0.0,
        "Hit Rate": (ic > 0).mean(),
        "t-stat": ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0.0,
        "Max IC": ic.max(),
        "Min IC": ic.min(),
        "N Months": len(ic),
    }


def ic_decay_analysis(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    max_lag: int = 12,
    end_date: str = None,
) -> pd.DataFrame:
    """IC at different holding horizons to measure signal persistence."""
    decay = {}
    for lag in range(1, max_lag + 1):
        shifted_returns = returns.shift(-(lag - 1))
        ic_series = compute_ic_series(signal, shifted_returns, universe, end_date=end_date)
        summary = ic_summary(ic_series)
        if summary:
            decay[lag] = {"Mean IC": summary["Mean IC"], "ICIR": summary["ICIR"]}
    return pd.DataFrame(decay).T
