"""Auto-generated signal: icaptq_div_lctq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IcaptqDivLctq(Factor):
    """icaptq / lctq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.238, Hit Rate=0.434,
        Turnover=0.016, Spread t=-2.80
    Validation ICIR (2005-2014): -0.114
    """

    name = "icaptq_div_lctq"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("icaptq")
        b = panel.pivot("lctq")
        return remove_infinities(a / b)
