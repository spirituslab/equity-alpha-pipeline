"""Auto-generated signal: oibdpq_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OibdpqToMktcap(Factor):
    """oibdpq / market_cap

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.266, Hit Rate=0.602,
        Turnover=0.083, Spread t=4.19
    Validation ICIR (2005-2014): 0.188
    """

    name = "oibdpq_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("oibdpq")
        mktcap = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities(field / mktcap)
