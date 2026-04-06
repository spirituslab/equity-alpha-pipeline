"""Ken French factor library loader."""

import pandas as pd
import pandas_datareader.data as web


def load_french_factors(start: str = "1962-01", end: str = "2020-01") -> pd.DataFrame:
    """Load Fama-French 5 factors + Momentum (monthly).

    Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    Index: PeriodIndex (monthly)
    Values: decimal returns (divided by 100 from raw)
    """
    # FF5 factors
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start, end=end)
    ff5_monthly = ff5[0]  # first table is monthly

    # Momentum factor
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start=start, end=end)
    mom_monthly = mom[0]

    # Combine
    factors = ff5_monthly.copy()
    factors["Mom"] = mom_monthly.iloc[:, 0]  # first column is Mom

    # Convert from percentage to decimal
    factors = factors / 100.0

    # Ensure PeriodIndex
    if not isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_period("M")

    return factors


def load_french_industry(n_industries: int = 12) -> dict[int, str]:
    """Load Ken French industry classification.

    Returns dict mapping SIC code ranges to industry names.
    This is used as a fallback for sector classification.
    """
    dataset_name = f"{n_industries}_Industry_Portfolios"
    data = web.DataReader(dataset_name, "famafrench")
    # The industry definitions are embedded in the dataset description
    # For practical use, we map via SIC code ranges
    return data
