"""Auto-generated signal: oancfy_div_acoq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivAcoq(Factor):
    """oancfy / acoq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.284, Hit Rate=0.628,
        Turnover=0.118, Spread t=3.22
    Validation ICIR (2005-2014): 0.221
    """

    name = "oancfy_div_acoq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("acoq")
        return remove_infinities(a / b)
