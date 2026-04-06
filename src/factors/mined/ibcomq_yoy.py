"""Auto-generated signal: ibcomq_yoy"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqYoy(Factor):
    """YoY growth of ibcomq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.311, Hit Rate=0.632,
        Turnover=0.217, Spread t=3.72
    Validation ICIR (2005-2014): 0.159
    """

    name = "ibcomq_yoy"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("ibcomq")
        return remove_infinities(field / field.shift(12) - 1)
