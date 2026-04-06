"""Auto-generated signal: cshoq_to_sales"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class CshoqToSales(Factor):
    """cshoq / sales

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.176, Hit Rate=0.393,
        Turnover=0.011, Spread t=-2.12
    Validation ICIR (2005-2014): -0.145
    """

    name = "cshoq_to_sales"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("cshoq")
        saleq = panel.pivot("saleq")
        return remove_infinities(field / saleq)
