"""Auto-generated signal: lctq_div_seqq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class LctqDivSeqq(Factor):
    """lctq / seqq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.195, Hit Rate=0.607,
        Turnover=0.020, Spread t=2.28
    Validation ICIR (2005-2014): 0.138
    """

    name = "lctq_div_seqq"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("lctq")
        b = panel.pivot("seqq")
        return remove_infinities(a / b)
