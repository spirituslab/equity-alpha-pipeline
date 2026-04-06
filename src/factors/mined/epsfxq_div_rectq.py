"""Auto-generated signal: epsfxq_div_rectq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqDivRectq(Factor):
    """epsfxq / rectq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.196, Hit Rate=0.569,
        Turnover=0.075, Spread t=2.22
    Validation ICIR (2005-2014): 0.254
    """

    name = "epsfxq_div_rectq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("epsfxq")
        b = panel.pivot("rectq")
        return remove_infinities(a / b)
