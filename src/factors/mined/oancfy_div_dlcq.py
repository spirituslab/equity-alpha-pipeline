"""Auto-generated signal: oancfy_div_dlcq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivDlcq(Factor):
    """oancfy / dlcq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.294, Hit Rate=0.607,
        Turnover=0.112, Spread t=4.31
    Validation ICIR (2005-2014): 0.273
    """

    name = "oancfy_div_dlcq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("dlcq")
        return remove_infinities(a / b)
