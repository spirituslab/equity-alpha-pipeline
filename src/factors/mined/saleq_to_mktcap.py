"""Auto-generated signal: saleq_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class SaleqToMktcap(Factor):
    """saleq / market_cap

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.164, Hit Rate=0.586,
        Turnover=0.019, Spread t=2.66
    Validation ICIR (2005-2014): 0.132
    """

    name = "saleq_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("saleq")
        mktcap = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities(field / mktcap)
