"""Auto-generated signal: oibdpq_yoy"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OibdpqYoy(Factor):
    """YoY growth of oibdpq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.227, Hit Rate=0.600,
        Turnover=0.152, Spread t=3.67
    Validation ICIR (2005-2014): 0.163
    """

    name = "oibdpq_yoy"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("oibdpq")
        return remove_infinities(field / field.shift(12) - 1)
