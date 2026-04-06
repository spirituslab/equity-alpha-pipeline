"""Auto-generated signal: rectq_qoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class RectqQoq(Factor):
    """QoQ growth of rectq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.198, Hit Rate=0.428,
        Turnover=0.369, Spread t=-3.44
    Validation ICIR (2005-2014): -0.254
    """

    name = "rectq_qoq"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("rectq")
        return remove_infinities(field / field.shift(3) - 1)
