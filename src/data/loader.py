"""Data loading and panel construction from CRSP/Compustat."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig


def load_compustat(config: PipelineConfig) -> pd.DataFrame:
    """Load Compustat/CRSP monthly panel data."""
    filepath = config.raw_path(config.data.compustat_file)

    str_cols = {"gvkey", "iid", "tic", "cusip", "TICKER", "lpermno"}
    dtype_map = {col: str for col in str_cols}
    df = pd.read_csv(filepath, dtype=dtype_map, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.to_period("M")
    df["sp500"] = df["sp500"].fillna(0).astype(int)
    df["gvkey"] = df["gvkey"].astype(str).str.strip()

    return df


def load_sp500_returns(config: PipelineConfig) -> pd.DataFrame:
    """Load S&P 500 index returns and risk-free rate."""
    filepath = config.raw_path(config.data.sp500_file)
    df = pd.read_csv(filepath, dtype=str)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m").dt.to_period("M")
    df = df.set_index("Date")
    df["ret_sp500"] = df["ret_sp500"].astype(float)
    df["rf"] = df["rf"].astype(float)
    df["excess_return"] = df["ret_sp500"] - df["rf"]
    return df


class DataPanel:
    """Central data panel providing pivoted views of the CRSP/Compustat data.

    Adapted from existing project with new methods for:
    - Market cap computation
    - Beta estimation
    - Universe filtering
    """

    QUARTERLY_FIELDS = {
        "epsfxq", "ceqq", "cshoq", "atq", "ltq", "dlcq", "dlttq",
        "saleq", "cogsq", "ibq", "dvpsxq", "oancfy", "cheq", "rectq",
        "invtq", "acoq",
        # Extended for signal mining
        "apq", "seqq", "oibdpq", "ibcomq", "pstkq", "icaptq", "mibq",
        "capxy", "ivncfy", "dvy", "lctq",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._raw = None
        self._sp500 = None

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            self._raw = load_compustat(self.config)
        return self._raw

    @property
    def sp500(self) -> pd.DataFrame:
        if self._sp500 is None:
            self._sp500 = load_sp500_returns(self.config)
        return self._sp500

    def pivot(self, field: str) -> pd.DataFrame:
        """Pivot raw data into (date x gvkey) matrix.

        Quarterly fields are forward-filled up to 3 months (no look-ahead).
        """
        start, end = self.config.dates.start, self.config.dates.end
        pivoted = self.raw.pivot_table(index="date", columns="gvkey", values=field, aggfunc="first")
        pivoted = pivoted.loc[start:end]

        if field in self.QUARTERLY_FIELDS:
            pivoted = pivoted.ffill(limit=3)

        return pivoted

    def get_returns(self, extend_end: str = "2020-01") -> pd.DataFrame:
        """Stock returns matrix (decimal). Extends one month past end for t+1 lookups."""
        pivoted = self.raw.pivot_table(index="date", columns="gvkey", values="trt1m", aggfunc="first")
        return pivoted.loc[self.config.dates.start:extend_end] / 100.0

    def get_sp500_membership(self) -> pd.DataFrame:
        """S&P 500 membership as (date x gvkey) boolean DataFrame."""
        start, end = self.config.dates.start, self.config.dates.end
        is_sp500 = self.raw.pivot_table(index="date", columns="gvkey", values="sp500", aggfunc="first")
        return is_sp500.loc[start:end] == 1

    def get_market_cap(self) -> pd.DataFrame:
        """Market cap in millions: prccm * cshoq."""
        prccm = self.pivot("prccm")
        cshoq = self.pivot("cshoq")
        return prccm * cshoq

    def get_risk_free(self) -> pd.Series:
        """Monthly risk-free rate series."""
        start, end = self.config.dates.start, self.config.dates.end
        return self.sp500["rf"].loc[start:end]

    def get_market_excess(self) -> pd.Series:
        """Monthly market excess return series."""
        start, end = self.config.dates.start, self.config.dates.end
        return self.sp500["excess_return"].loc[start:end]

    def get_universe(self) -> pd.DataFrame:
        """Investable universe mask (date x gvkey) boolean DataFrame.

        Criteria:
        - S&P 500 member at that date
        - Valid price
        - Market cap > min_mcap_mm
        - At least min_history_months of return history
        """
        sp500 = self.get_sp500_membership()
        prccm = self.pivot("prccm")
        mcap = self.get_market_cap()
        returns = self.get_returns()

        has_price = prccm.notna()
        has_mcap = mcap > self.config.universe.min_mcap_mm
        has_history = returns.rolling(self.config.universe.min_history_months).count() >= self.config.universe.min_history_months

        universe = sp500 & has_price & has_mcap & has_history

        # Align indices
        common_dates = universe.index.intersection(sp500.index)
        common_stocks = universe.columns.intersection(sp500.columns)
        return universe.loc[common_dates, common_stocks]

    def get_log_market_cap(self) -> pd.DataFrame:
        """Log market cap for neutralization."""
        mcap = self.get_market_cap()
        return np.log(mcap.replace(0, np.nan))

    def get_rolling_beta(self, window: int = 60) -> pd.DataFrame:
        """Rolling market beta for each stock.

        Beta estimated from rolling regression of stock excess returns
        on market excess returns over trailing window.
        """
        returns = self.get_returns()
        mkt = self.get_market_excess()
        rf = self.get_risk_free()

        # Align
        common_dates = returns.index.intersection(mkt.index).intersection(rf.index)
        returns = returns.loc[common_dates]
        mkt = mkt.loc[common_dates]
        rf = rf.loc[common_dates]

        beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for i in range(window, len(returns)):
            t = returns.index[i]
            r_win = returns.iloc[i - window:i]
            mkt_win = mkt.iloc[i - window:i].values
            rf_win = rf.iloc[i - window:i].values

            mkt_var = np.var(mkt_win)
            if mkt_var == 0:
                continue

            for stock in r_win.columns:
                y = r_win[stock].values - rf_win
                valid = ~np.isnan(y)
                if valid.sum() < 24:
                    continue
                cov = np.cov(y[valid], mkt_win[valid])[0, 1]
                var = np.var(mkt_win[valid])
                if var > 0:
                    beta.loc[t, stock] = cov / var

        return beta
