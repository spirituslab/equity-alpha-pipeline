"""Auto-generated signal: cheq_qoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class CheqQoq(Factor):
    """QoQ growth of cheq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.255, Hit Rate=0.593,
        Turnover=0.400, Spread t=2.68
    Validation ICIR (2005-2014): 0.351
    """

    name = "cheq_qoq"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("cheq")
        return remove_infinities(field / field.shift(3) - 1)
