"""Auto-generated signal: epsfxq_to_equity"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqToEquity(Factor):
    """epsfxq / common_equity

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.212, Hit Rate=0.586,
        Turnover=0.087, Spread t=3.12
    Validation ICIR (2005-2014): 0.270
    """

    name = "epsfxq_to_equity"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("epsfxq")
        ceqq = panel.pivot("ceqq")
        return remove_infinities(field / ceqq)
