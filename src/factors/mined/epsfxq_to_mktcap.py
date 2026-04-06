"""Auto-generated signal: epsfxq_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqToMktcap(Factor):
    """epsfxq / market_cap

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.293, Hit Rate=0.623,
        Turnover=0.098, Spread t=4.47
    Validation ICIR (2005-2014): 0.333
    """

    name = "epsfxq_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("epsfxq")
        mktcap = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities(field / mktcap)
