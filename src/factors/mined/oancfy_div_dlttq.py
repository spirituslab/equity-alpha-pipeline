"""Auto-generated signal: oancfy_div_dlttq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivDlttq(Factor):
    """oancfy / dlttq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.384, Hit Rate=0.652,
        Turnover=0.109, Spread t=3.81
    Validation ICIR (2005-2014): 0.280
    """

    name = "oancfy_div_dlttq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("dlttq")
        return remove_infinities(a / b)
