"""Auto-generated signal: epsfxq_div_ltq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqDivLtq(Factor):
    """epsfxq / ltq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.183, Hit Rate=0.571,
        Turnover=0.071, Spread t=2.36
    Validation ICIR (2005-2014): 0.206
    """

    name = "epsfxq_div_ltq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("epsfxq")
        b = panel.pivot("ltq")
        return remove_infinities(a / b)
