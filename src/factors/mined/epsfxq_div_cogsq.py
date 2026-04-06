"""Auto-generated signal: epsfxq_div_cogsq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqDivCogsq(Factor):
    """epsfxq / cogsq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.203, Hit Rate=0.579,
        Turnover=0.070, Spread t=2.58
    Validation ICIR (2005-2014): 0.140
    """

    name = "epsfxq_div_cogsq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("epsfxq")
        b = panel.pivot("cogsq")
        return remove_infinities(a / b)
