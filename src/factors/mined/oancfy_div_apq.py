"""Auto-generated signal: oancfy_div_apq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivApq(Factor):
    """oancfy / apq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.281, Hit Rate=0.621,
        Turnover=0.088, Spread t=3.81
    Validation ICIR (2005-2014): 0.195
    """

    name = "oancfy_div_apq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("apq")
        return remove_infinities(a / b)
