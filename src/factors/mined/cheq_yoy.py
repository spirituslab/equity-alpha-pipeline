"""Auto-generated signal: cheq_yoy"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class CheqYoy(Factor):
    """YoY growth of cheq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.203, Hit Rate=0.604,
        Turnover=0.152, Spread t=2.51
    Validation ICIR (2005-2014): 0.232
    """

    name = "cheq_yoy"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("cheq")
        return remove_infinities(field / field.shift(12) - 1)
