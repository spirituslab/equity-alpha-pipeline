"""Auto-generated signal: oibdpq_div_icaptq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OibdpqDivIcaptq(Factor):
    """oibdpq / icaptq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.169, Hit Rate=0.566,
        Turnover=0.067, Spread t=2.91
    Validation ICIR (2005-2014): 0.256
    """

    name = "oibdpq_div_icaptq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oibdpq")
        b = panel.pivot("icaptq")
        return remove_infinities(a / b)
