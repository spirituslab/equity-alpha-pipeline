"""Auto-generated signal: lctq_div_icaptq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class LctqDivIcaptq(Factor):
    """lctq / icaptq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.235, Hit Rate=0.572,
        Turnover=0.017, Spread t=2.93
    Validation ICIR (2005-2014): 0.123
    """

    name = "lctq_div_icaptq"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("lctq")
        b = panel.pivot("icaptq")
        return remove_infinities(a / b)
