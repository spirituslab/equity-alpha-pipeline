"""Auto-generated signal: ibcomq_div_acoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqDivAcoq(Factor):
    """ibcomq / acoq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.220, Hit Rate=0.586,
        Turnover=0.091, Spread t=2.29
    Validation ICIR (2005-2014): 0.149
    """

    name = "ibcomq_div_acoq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("ibcomq")
        b = panel.pivot("acoq")
        return remove_infinities(a / b)
