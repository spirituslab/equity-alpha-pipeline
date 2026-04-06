import math

import numpy as np
import pandas as pd


def _norm_ppf(p):
    """Normal distribution inverse CDF without scipy."""
    if p <= 0 or p >= 1:
        return float("nan")
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric VaR assuming normal distribution."""
    z = _norm_ppf(1 - confidence)
    return returns.mean() + z * returns.std()


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (percentile-based)."""
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Cornish-Fisher VaR adjusting for skewness and kurtosis."""
    r = returns.dropna()
    z = _norm_ppf(1 - confidence)
    s = pd.Series(r).skew()
    k = pd.Series(r).kurtosis()
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return returns.mean() + z_cf * returns.std()


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown time series."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def drawdown_stats(returns: pd.Series) -> dict:
    """Max drawdown, avg drawdown, max duration."""
    dd = drawdown_series(returns)
    max_dd = dd.min()
    in_drawdown = dd < 0
    duration = 0
    max_duration = 0
    for val in in_drawdown:
        if val:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return {
        "Max Drawdown": max_dd,
        "Avg Drawdown": dd[dd < 0].mean() if (dd < 0).any() else 0.0,
        "Max Drawdown Duration (months)": max_duration,
    }
