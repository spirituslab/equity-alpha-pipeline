import math

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annualize: bool = True) -> float:
    """Annualized Sharpe ratio from monthly returns."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    return sr * np.sqrt(12) if annualize else sr


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """Annualized Sortino ratio using downside deviation (monthly data)."""
    excess = returns - target
    downside = returns[returns < target]
    downside_std = downside.std()
    if downside_std <= 0:
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(12)


def calmar_ratio(returns: pd.Series) -> float:
    """Annualized return / abs(max drawdown)."""
    ann_ret = returns.mean() * 12
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a return series (decimal returns)."""
    r = returns.copy()
    if r.abs().mean() > 1:
        r = r / 100.0
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative returns from simple return series."""
    return (1 + returns).cumprod()


def rolling_sharpe(returns: pd.Series, window: int = 36) -> pd.Series:
    """Rolling Sharpe ratio (annualized)."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(12)


def compute_descriptive_stats(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Descriptive statistics for return series."""
    stats = {}
    for col in returns.columns:
        ret = returns[col].dropna()
        if len(ret) == 0:
            continue
        mean = ret.mean()
        std = ret.std()
        sr = mean / std if std > 0 else 0.0
        if annualize:
            mean *= 12
            std *= np.sqrt(12)
            sr *= np.sqrt(12)
        stats[col] = {
            "Ann. Return": mean,
            "Ann. Vol": std,
            "Sharpe": sr,
            "Sortino": sortino_ratio(ret),
            "Calmar": calmar_ratio(ret),
            "Max Drawdown": max_drawdown(ret),
            "Skewness": pd.Series(ret).skew(),
            "Kurtosis": pd.Series(ret).kurtosis(),
            "Win Rate": (ret > 0).mean(),
        }
    return pd.DataFrame(stats)


def t_test_mean(returns: pd.Series) -> dict:
    """Test whether mean return is significantly different from zero."""
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {"t_stat": 0, "p_value": 1.0}
    mean = r.mean()
    se = r.std() / np.sqrt(n)
    t_stat = mean / se if se > 0 else 0.0
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return {"t_stat": t_stat, "p_value": p_value}
