"""Auto-generated signal: oancfy_div_oibdpq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivOibdpq(Factor):
    """oancfy / oibdpq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.332, Hit Rate=0.637,
        Turnover=0.202, Spread t=5.63
    Validation ICIR (2005-2014): 0.196
    """

    name = "oancfy_div_oibdpq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("oibdpq")
        return remove_infinities(a / b)
