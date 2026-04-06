"""Auto-generated signal: saleq_minus_cogsq_div_ceqq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class SaleqMinusCogsqDivCeqq(Factor):
    """(saleq - cogsq) / ceqq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.247, Hit Rate=0.566,
        Turnover=0.036, Spread t=3.52
    Validation ICIR (2005-2014): 0.198
    """

    name = "saleq_minus_cogsq_div_ceqq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("saleq")
        b = panel.pivot("cogsq")
        c = panel.pivot("ceqq")
        return remove_infinities((a - b) / c)
