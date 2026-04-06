"""Factor attribution via time-series regression on FF5 + Momentum.

r_{p,t} - rf_t = alpha + b1*MktRF + b2*SMB + b3*HML + b4*RMW + b5*CMA + b6*Mom + epsilon

A statistically significant positive alpha means the strategy generates
returns beyond what FF factors explain — true stock selection skill.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    rf: pd.Series = None,
) -> dict:
    """Time-series regression of portfolio returns on FF5 + Momentum.

    Args:
        portfolio_returns: monthly strategy returns (gross or net)
        factor_returns: (T x K) Fama-French factor returns (includes Mkt-RF)
        rf: risk-free rate. If provided, subtracts from portfolio returns.

    Returns:
        dict with alpha, factor loadings, t-stats, R-squared
    """
    # Align dates
    common = portfolio_returns.dropna().index.intersection(factor_returns.dropna(how="all").index)
    if rf is not None:
        common = common.intersection(rf.dropna().index)

    y = portfolio_returns.loc[common]
    X = factor_returns.loc[common]

    if rf is not None:
        y = y - rf.loc[common]

    # Drop any remaining NaN rows
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]

    if len(y) < 12:
        return {"error": "Not enough observations for regression"}

    # OLS with Newey-West HAC standard errors
    X_const = sm.add_constant(X)
    model = sm.OLS(y.values, X_const.values).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    # Extract results
    factor_names = ["alpha"] + X.columns.tolist()
    result = {
        "alpha": model.params[0] * 12,  # annualized
        "alpha_t_stat": model.tvalues[0],
        "alpha_p_value": model.pvalues[0],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
    }

    # Factor loadings
    for i, name in enumerate(X.columns):
        result[f"beta_{name}"] = model.params[i + 1]
        result[f"t_{name}"] = model.tvalues[i + 1]
        result[f"p_{name}"] = model.pvalues[i + 1]

    return result


def print_attribution(result: dict) -> None:
    """Pretty-print factor attribution results."""
    print(f"\n{'='*60}")
    print(f"  FACTOR ATTRIBUTION (FF5 + Momentum)")
    print(f"{'='*60}")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    print(f"\n  Alpha (annualized): {result['alpha']:>8.4f}  "
          f"(t={result['alpha_t_stat']:>6.2f}, p={result['alpha_p_value']:>6.4f})")
    sig = "***" if result["alpha_p_value"] < 0.01 else "**" if result["alpha_p_value"] < 0.05 else "*" if result["alpha_p_value"] < 0.10 else ""
    print(f"  Significance: {sig}")
    print(f"  R-squared:          {result['r_squared']:>8.4f}")
    print(f"  Observations:       {result['n_obs']:>8d}")

    print(f"\n  Factor Loadings:")
    for key in result:
        if key.startswith("beta_") and not key.startswith("beta_exposure"):
            name = key.replace("beta_", "")
            t_key = f"t_{name}"
            print(f"    {name:10s}: {result[key]:>8.4f}  (t={result.get(t_key, 0):>6.2f})")

    print(f"{'='*60}\n")
