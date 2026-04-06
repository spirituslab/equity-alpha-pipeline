"""Auto-generated signal: ibq_minus_oancfy_div_atq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbqMinusOancfyDivAtq(Factor):
    """(ibq - oancfy) / atq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.434, Hit Rate=0.310,
        Turnover=0.183, Spread t=-5.53
    Validation ICIR (2005-2014): -0.273
    """

    name = "ibq_minus_oancfy_div_atq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("ibq")
        b = panel.pivot("oancfy")
        c = panel.pivot("atq")
        return remove_infinities((a - b) / c)
