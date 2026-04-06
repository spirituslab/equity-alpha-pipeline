"""Auto-generated signal: ibcomq_qoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqQoq(Factor):
    """QoQ growth of ibcomq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.204, Hit Rate=0.578,
        Turnover=0.373, Spread t=2.66
    Validation ICIR (2005-2014): 0.121
    """

    name = "ibcomq_qoq"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("ibcomq")
        return remove_infinities(field / field.shift(3) - 1)
