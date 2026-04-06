"""Auto-generated signal: dlttq_yoy"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class DlttqYoy(Factor):
    """YoY growth of dlttq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.153, Hit Rate=0.445,
        Turnover=0.090, Spread t=-3.46
    Validation ICIR (2005-2014): -0.139
    """

    name = "dlttq_yoy"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("dlttq")
        return remove_infinities(field / field.shift(12) - 1)
