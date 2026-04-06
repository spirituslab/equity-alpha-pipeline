"""Auto-generated signal: ivncfy_div_oibdpq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IvncfyDivOibdpq(Factor):
    """ivncfy / oibdpq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.234, Hit Rate=0.634,
        Turnover=0.156, Spread t=2.17
    Validation ICIR (2005-2014): 0.213
    """

    name = "ivncfy_div_oibdpq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("ivncfy")
        b = panel.pivot("oibdpq")
        return remove_infinities(a / b)
