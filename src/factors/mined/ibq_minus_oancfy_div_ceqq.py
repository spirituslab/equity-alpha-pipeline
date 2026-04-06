"""Auto-generated signal: ibq_minus_oancfy_div_ceqq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbqMinusOancfyDivCeqq(Factor):
    """(ibq - oancfy) / ceqq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=-0.392, Hit Rate=0.297,
        Turnover=0.178, Spread t=-4.55
    Validation ICIR (2005-2014): -0.239
    """

    name = "ibq_minus_oancfy_div_ceqq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("ibq")
        b = panel.pivot("oancfy")
        c = panel.pivot("ceqq")
        return remove_infinities((a - b) / c)
