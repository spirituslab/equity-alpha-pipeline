"""Auto-generated signal: epsfxq_div_dlcq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqDivDlcq(Factor):
    """epsfxq / dlcq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.207, Hit Rate=0.593,
        Turnover=0.082, Spread t=2.29
    Validation ICIR (2005-2014): 0.222
    """

    name = "epsfxq_div_dlcq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("epsfxq")
        b = panel.pivot("dlcq")
        return remove_infinities(a / b)
