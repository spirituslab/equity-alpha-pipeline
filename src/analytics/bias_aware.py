"""Selection-bias-aware statistics for backtest evaluation.

Implements:
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Probability of Backtest Overfitting (Bailey et al., 2015)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_months: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio — probability that observed SR exceeds zero
    after correcting for multiple testing.

    Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"

    Args:
        observed_sharpe: annualized Sharpe ratio of selected strategy
        n_trials: number of strategies tested (e.g., 770 candidates)
        n_months: number of months in evaluation period
        skewness: skewness of monthly returns
        excess_kurtosis: excess kurtosis of monthly returns

    Returns:
        DSR probability in [0, 1]. High = likely genuine, low = likely luck.
    """
    if n_trials <= 0 or n_months <= 0:
        return np.nan

    # Convert annualized Sharpe to monthly
    sr_monthly = observed_sharpe / np.sqrt(12)

    # Expected maximum Sharpe under the null (Euler-Mascheroni approximation)
    # E[max(SR)] ≈ sqrt(V[SR]) * ((1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e)))
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    gamma_em = 0.5772156649
    e = np.exp(1)

    # Variance of SR estimator: V[SR] ≈ (1 + 0.5*SR²) / T (simplified)
    # More precise: accounts for skew and kurtosis
    sr_var = (1 - skewness * sr_monthly + (excess_kurtosis / 4) * sr_monthly ** 2) / n_months

    if sr_var <= 0:
        return np.nan

    sr_std = np.sqrt(sr_var)

    # Expected max SR under null of N independent trials
    if n_trials == 1:
        sr0 = 0.0
    else:
        z1 = stats.norm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0
        z2 = stats.norm.ppf(1 - 1 / (n_trials * e)) if n_trials * e > 1 else 0
        sr0 = sr_std * ((1 - gamma_em) * z1 + gamma_em * z2)

    # DSR = Φ((SR* - SR_0) / sqrt(V[SR]))
    if sr_std == 0:
        return np.nan

    z_score = (sr_monthly - sr0) / sr_std
    dsr = float(stats.norm.cdf(z_score))

    return dsr


def probability_of_backtest_overfitting(
    fold_performance: pd.DataFrame,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (CSCV).

    Bailey et al. (2015): "The Probability of Backtest Overfitting"

    Args:
        fold_performance: DataFrame with shape (n_strategies, n_folds).
            Each cell is the Sharpe ratio of strategy i on fold j.

    Returns:
        dict with:
            pbo: probability of overfitting [0, 1]
            logit_distribution: array of logit values per partition
            n_partitions: number of CSCV partitions tested
    """
    n_strategies, n_folds = fold_performance.shape

    if n_folds < 2 or n_strategies < 2:
        return {"pbo": np.nan, "logit_distribution": np.array([]), "n_partitions": 0}

    # Ensure even number of folds for symmetric split
    half = n_folds // 2
    fold_indices = list(range(n_folds))

    # Generate all C(n_folds, half) train/test partitions
    logits = []
    for train_folds in combinations(fold_indices, half):
        test_folds = [f for f in fold_indices if f not in train_folds]

        # In-sample performance: average across train folds
        is_perf = fold_performance.iloc[:, list(train_folds)].mean(axis=1)

        # Out-of-sample performance: average across test folds
        oos_perf = fold_performance.iloc[:, test_folds].mean(axis=1)

        # Rank strategies by IS performance (best = highest rank)
        is_rank = is_perf.rank(ascending=True)

        # Find the IS-best strategy
        is_best_idx = is_perf.idxmax()

        # What is the OOS rank of the IS-best strategy?
        oos_rank = oos_perf.rank(ascending=True)
        oos_rank_of_best = oos_rank[is_best_idx]

        # Relative OOS rank: 0 = worst, 1 = best
        relative_rank = oos_rank_of_best / n_strategies

        # Logit of relative rank
        if relative_rank <= 0 or relative_rank >= 1:
            # Clamp to avoid log(0)
            relative_rank = np.clip(relative_rank, 0.01, 0.99)

        logit = np.log(relative_rank / (1 - relative_rank))
        logits.append(logit)

    logits = np.array(logits)

    # PBO = fraction of partitions where IS-best underperforms OOS median
    # (logit < 0 means below-median OOS performance)
    pbo = float((logits <= 0).mean())

    return {
        "pbo": pbo,
        "logit_distribution": logits,
        "n_partitions": len(logits),
    }


def bias_aware_report(
    oos_returns: pd.Series,
    n_trials: int,
    fold_performance: pd.DataFrame = None,
) -> dict:
    """Compute all bias-aware statistics for a strategy.

    Args:
        oos_returns: monthly OOS return series
        n_trials: number of strategies tested during research
        fold_performance: (n_strategies × n_folds) Sharpe matrix for PBO

    Returns:
        dict with DSR, PBO, and descriptive statistics
    """
    r = oos_returns.dropna()
    n_months = len(r)

    if n_months < 12:
        return {"dsr": np.nan, "pbo": np.nan, "n_trials": n_trials}

    observed_sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))  # excess kurtosis

    dsr = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_months=n_months,
        skewness=skew,
        excess_kurtosis=kurt,
    )

    result = {
        "observed_sharpe": observed_sharpe,
        "dsr": dsr,
        "n_trials": n_trials,
        "n_months": n_months,
        "skewness": skew,
        "excess_kurtosis": kurt,
    }

    if fold_performance is not None:
        pbo_result = probability_of_backtest_overfitting(fold_performance)
        result["pbo"] = pbo_result["pbo"]
        result["pbo_n_partitions"] = pbo_result["n_partitions"]
    else:
        result["pbo"] = np.nan

    return result
