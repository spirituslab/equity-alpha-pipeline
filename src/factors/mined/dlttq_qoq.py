"""Auto-generated signal: dlttq_qoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class DlttqQoq(Factor):
    """QoQ growth of dlttq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.167, Hit Rate=0.434,
        Turnover=0.310, Spread t=-2.02
    Validation ICIR (2005-2014): -0.194
    """

    name = "dlttq_qoq"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("dlttq")
        return remove_infinities(field / field.shift(3) - 1)
