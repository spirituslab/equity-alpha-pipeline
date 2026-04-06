"""Auto-generated signal: ibcomq_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqToMktcap(Factor):
    """ibcomq / market_cap

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.468, Hit Rate=0.691,
        Turnover=0.169, Spread t=6.76
    Validation ICIR (2005-2014): 0.302
    """

    name = "ibcomq_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("ibcomq")
        mktcap = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities(field / mktcap)
