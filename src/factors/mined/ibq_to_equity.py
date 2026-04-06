"""Auto-generated signal: ibq_to_equity"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbqToEquity(Factor):
    """ibq / common_equity

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.175, Hit Rate=0.586,
        Turnover=0.126, Spread t=2.58
    Validation ICIR (2005-2014): 0.226
    """

    name = "ibq_to_equity"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("ibq")
        ceqq = panel.pivot("ceqq")
        return remove_infinities(field / ceqq)
