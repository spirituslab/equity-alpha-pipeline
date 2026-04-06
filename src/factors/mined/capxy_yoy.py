"""Auto-generated signal: capxy_yoy"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class CapxyYoy(Factor):
    """YoY growth of capxy

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.204, Hit Rate=0.414,
        Turnover=0.102, Spread t=-2.12
    Validation ICIR (2005-2014): -0.179
    """

    name = "capxy_yoy"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("capxy")
        return remove_infinities(field / field.shift(12) - 1)
