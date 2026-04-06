"""Block bootstrap for confidence intervals on performance metrics.

Uses circular block bootstrap (Politis & Romano 1992) to preserve
autocorrelation structure in monthly returns.
"""

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    block_size: int = 12,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Block bootstrap for Sharpe ratio confidence intervals.

    Args:
        returns: monthly return series
        n_bootstrap: number of bootstrap samples
        block_size: block size (12 = 1 year, preserves annual autocorrelation)
        confidence: confidence level for CI
        seed: random seed

    Returns:
        dict with point estimate, CI bounds, and standard error
    """
    r = returns.dropna().values
    n = len(r)

    if n < block_size * 2:
        return {"sharpe": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    rng = np.random.default_rng(seed)

    # Point estimate
    point_sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0

    # Bootstrap
    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Circular block bootstrap
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        boot_sample = r[indices]
        std = boot_sample.std()
        boot_sharpes[b] = boot_sample.mean() / std * np.sqrt(12) if std > 0 else 0

    alpha = 1 - confidence
    ci_lower = np.percentile(boot_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(boot_sharpes, (1 - alpha / 2) * 100)

    return {
        "sharpe": point_sharpe,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": boot_sharpes.std(),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }


def block_bootstrap_metric(
    returns: pd.Series,
    metric_fn,
    n_bootstrap: int = 10000,
    block_size: int = 12,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Block bootstrap for any metric function.

    Args:
        returns: monthly return series
        metric_fn: callable(pd.Series) -> float
        n_bootstrap, block_size, confidence, seed: same as above
    """
    r = returns.dropna()
    n = len(r)

    if n < block_size * 2:
        return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    rng = np.random.default_rng(seed)
    point = metric_fn(r)

    n_blocks = int(np.ceil(n / block_size))
    boot_values = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        boot_sample = pd.Series(r.values[indices])
        boot_values[b] = metric_fn(boot_sample)

    alpha = 1 - confidence
    return {
        "point": point,
        "ci_lower": np.percentile(boot_values, alpha / 2 * 100),
        "ci_upper": np.percentile(boot_values, (1 - alpha / 2) * 100),
        "se": boot_values.std(),
    }
