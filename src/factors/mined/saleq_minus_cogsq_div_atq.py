"""Auto-generated signal: saleq_minus_cogsq_div_atq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class SaleqMinusCogsqDivAtq(Factor):
    """(saleq - cogsq) / atq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.165, Hit Rate=0.559,
        Turnover=0.024, Spread t=2.32
    Validation ICIR (2005-2014): 0.178
    """

    name = "saleq_minus_cogsq_div_atq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("saleq")
        b = panel.pivot("cogsq")
        c = panel.pivot("atq")
        return remove_infinities((a - b) / c)
