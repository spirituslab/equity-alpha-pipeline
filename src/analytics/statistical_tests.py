import math

import numpy as np
import pandas as pd


def diebold_mariano_test(
    returns_1: pd.Series,
    returns_2: pd.Series,
    benchmark_returns: pd.Series = None,
    loss: str = "SE",
) -> dict:
    """Diebold-Mariano test for comparing two strategies."""
    common = np.intersect1d(returns_1.dropna().index.values, returns_2.dropna().index.values)
    r1 = returns_1.loc[common]
    r2 = returns_2.loc[common]

    if benchmark_returns is not None:
        bench = benchmark_returns.loc[common]
        e1 = r1 - bench
        e2 = r2 - bench
    else:
        e1 = r1
        e2 = r2

    if loss == "SE":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)

    n = len(d)
    d_bar = d.mean()
    d_var = d.var(ddof=1)
    dm_stat = d_bar / np.sqrt(d_var / n) if d_var > 0 else 0.0
    p_value = math.erfc(abs(dm_stat) / math.sqrt(2))

    return {
        "DM Statistic": dm_stat,
        "p-value": p_value,
        "Significant (5%)": p_value < 0.05,
        "N": n,
    }


def sharpe_ratio_test(returns_1: pd.Series, returns_2: pd.Series) -> dict:
    """Jobson-Korkie test with Memmel correction for Sharpe ratio equality."""
    common = np.intersect1d(returns_1.dropna().index.values, returns_2.dropna().index.values)
    r1 = returns_1.loc[common].values
    r2 = returns_2.loc[common].values
    n = len(r1)

    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    sr1 = mu1 / s1 if s1 > 0 else 0
    sr2 = mu2 / s2 if s2 > 0 else 0

    cov12 = np.cov(r1, r2)[0, 1]
    theta = (1 / n) * (
        2 * (1 - cov12 / (s1 * s2))
        + 0.5 * (sr1**2 + sr2**2 - sr1 * sr2 * (1 + (cov12 / (s1 * s2)) ** 2))
    )

    z_stat = (sr1 - sr2) / np.sqrt(theta) if theta > 0 else 0
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))

    return {
        "SR1": sr1,
        "SR2": sr2,
        "SR Diff": sr1 - sr2,
        "z-statistic": z_stat,
        "p-value": p_value,
        "Significant (5%)": p_value < 0.05,
    }
