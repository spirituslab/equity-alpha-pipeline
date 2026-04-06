"""Auto-generated signal: ibcomq_div_ltq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqDivLtq(Factor):
    """ibcomq / ltq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.201, Hit Rate=0.554,
        Turnover=0.076, Spread t=2.10
    Validation ICIR (2005-2014): 0.161
    """

    name = "ibcomq_div_ltq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("ibcomq")
        b = panel.pivot("ltq")
        return remove_infinities(a / b)
