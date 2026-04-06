import pandas as pd

from src.factors.base import Factor


class Momentum12_2(Factor):
    """12-2 Month Long-Term Momentum.

    Cumulative return from t-12 to t-2, skipping most recent month.
    Jegadeesh & Titman (1993): past winners continue to outperform.
    """

    name = "momentum_12_2"
    category = "momentum"

    def compute(self, panel) -> pd.DataFrame:
        trt1m = panel.pivot("trt1m")
        return trt1m.shift(2).rolling(window=11).mean()


class STReversal(Factor):
    """1-Month Short-Term Reversal.

    Negative of prior month return.
    Jegadeesh (1990): last month's losers outperform next month.
    """

    name = "st_reversal"
    category = "momentum"

    def compute(self, panel) -> pd.DataFrame:
        trt1m = panel.pivot("trt1m")
        return -trt1m
