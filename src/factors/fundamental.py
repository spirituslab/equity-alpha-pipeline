import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class ROE(Factor):
    """Return on Equity — Profitability.

    ROE = IBQ / CEQQ
    Fama-French RMW factor is related. High ROE = higher quality.
    """

    name = "roe"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        ceqq = panel.pivot("ceqq")
        return remove_infinities(ibq / ceqq)


class GrossProfitability(Factor):
    """Gross Profitability — Quality Factor.

    GP = (SALEQ - COGSQ) / ATQ
    Novy-Marx (2013): cleanest accounting measure of economic profitability.
    """

    name = "gross_profitability"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        saleq = panel.pivot("saleq")
        cogsq = panel.pivot("cogsq")
        atq = panel.pivot("atq")
        return remove_infinities((saleq - cogsq) / atq)


class AccrualRatio(Factor):
    """Accrual Ratio — Earnings Quality.

    AccrualRatio = (IBQ - OANCFY) / ATQ
    Sloan (1996): high accruals predict lower future returns.
    """

    name = "accrual_ratio"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        oancfy = panel.pivot("oancfy")
        atq = panel.pivot("atq")
        return remove_infinities((ibq - oancfy) / atq)


class AssetGrowth(Factor):
    """Asset Growth — Investment Factor.

    AssetGrowth = ATQ / lag(ATQ, 12) - 1
    Cooper, Gulen, Schill (2008): aggressive asset growth → lower returns.
    """

    name = "asset_growth"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        atq = panel.pivot("atq")
        return remove_infinities(atq / atq.shift(12) - 1)


class EarningsYield(Factor):
    """Earnings Yield — Valuation.

    EY = EPSFXQ / PRCCM
    Inverse of P/E. High EY = cheap stock.
    """

    name = "earnings_yield"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        epsfxq = panel.pivot("epsfxq")
        prccm = panel.pivot("prccm")
        return remove_infinities(epsfxq / prccm)
