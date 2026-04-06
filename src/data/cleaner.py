import numpy as np
import pandas as pd


def remove_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


def winsorize_cross_section(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """Winsorize each row (cross-section) at specified percentile limits."""
    result = df.copy()
    for idx in result.index:
        row = result.loc[idx].dropna()
        if len(row) > 0:
            lower = np.nanpercentile(row, limits[0] * 100)
            upper = np.nanpercentile(row, (1 - limits[1]) * 100)
            result.loc[idx] = result.loc[idx].clip(lower=lower, upper=upper)
    return result
