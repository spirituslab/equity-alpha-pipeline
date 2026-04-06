"""Auto-generated signal: invtq_qoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class InvtqQoq(Factor):
    """QoQ growth of invtq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.192, Hit Rate=0.421,
        Turnover=0.328, Spread t=-2.45
    Validation ICIR (2005-2014): -0.358
    """

    name = "invtq_qoq"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("invtq")
        return remove_infinities(field / field.shift(3) - 1)
