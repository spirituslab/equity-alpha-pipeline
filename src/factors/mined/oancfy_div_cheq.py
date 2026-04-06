"""Auto-generated signal: oancfy_div_cheq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivCheq(Factor):
    """oancfy / cheq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.200, Hit Rate=0.559,
        Turnover=0.094, Spread t=2.44
    Validation ICIR (2005-2014): 0.142
    """

    name = "oancfy_div_cheq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("cheq")
        return remove_infinities(a / b)
