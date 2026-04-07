"""Cross-sectional winsorization and z-scoring for signal standardization."""

import numpy as np
import pandas as pd


def winsorize_cross_section(
    signal: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Winsorize each cross-section (row) at specified percentiles.

    Clips extreme values to reduce impact of data errors, split artifacts,
    and filing anomalies on downstream neutralization and combination.
    """
    # Vectorized: compute percentiles across columns for each row
    vals = signal.values  # (T, N)
    lo = np.nanpercentile(vals, lower_pct * 100, axis=1, keepdims=True)
    hi = np.nanpercentile(vals, upper_pct * 100, axis=1, keepdims=True)
    clipped = np.clip(vals, lo, hi)
    # Preserve NaN
    clipped = np.where(np.isnan(vals), np.nan, clipped)
    return pd.DataFrame(clipped, index=signal.index, columns=signal.columns)


def zscore_cross_section(signal: pd.DataFrame) -> pd.DataFrame:
    """Convert raw signal to cross-sectional z-scores.

    For each date t:
        z_{i,t} = (x_{i,t} - mean_t) / std_t

    Makes different signals comparable in magnitude.
    """
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    # Avoid division by zero
    std = std.replace(0, np.nan)
    return signal.sub(mean, axis=0).div(std, axis=0)


def rank_cross_section(signal: pd.DataFrame) -> pd.DataFrame:
    """Convert to cross-sectional percentile ranks in [0, 1].

    More robust to outliers than z-scores. Useful as alternative
    standardization for ML inputs.
    """
    return signal.rank(axis=1, pct=True)


def standardize_signal(
    signal: pd.DataFrame,
    winsorize_pct: float = 0.01,
    method: str = "zscore",
) -> pd.DataFrame:
    """Full standardization pipeline: winsorize then standardize.

    Args:
        signal: (date x gvkey) raw factor exposures
        winsorize_pct: percentile for winsorization (e.g., 0.01 = 1st/99th)
        method: "zscore" or "rank"

    Returns:
        (date x gvkey) standardized signal values
    """
    # Step 1: Winsorize
    clipped = winsorize_cross_section(signal, lower_pct=winsorize_pct, upper_pct=1 - winsorize_pct)

    # Step 2: Standardize
    if method == "zscore":
        return zscore_cross_section(clipped)
    elif method == "rank":
        return rank_cross_section(clipped)
    else:
        raise ValueError(f"Unknown standardization method: {method}. Use 'zscore' or 'rank'.")
