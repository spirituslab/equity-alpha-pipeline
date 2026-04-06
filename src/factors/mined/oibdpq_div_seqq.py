"""Auto-generated signal: oibdpq_div_seqq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OibdpqDivSeqq(Factor):
    """oibdpq / seqq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.220, Hit Rate=0.586,
        Turnover=0.072, Spread t=2.11
    Validation ICIR (2005-2014): 0.222
    """

    name = "oibdpq_div_seqq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oibdpq")
        b = panel.pivot("seqq")
        return remove_infinities(a / b)
