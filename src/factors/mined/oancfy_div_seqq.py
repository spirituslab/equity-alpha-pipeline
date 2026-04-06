"""Auto-generated signal: oancfy_div_seqq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivSeqq(Factor):
    """oancfy / seqq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.401, Hit Rate=0.662,
        Turnover=0.142, Spread t=4.17
    Validation ICIR (2005-2014): 0.284
    """

    name = "oancfy_div_seqq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("seqq")
        return remove_infinities(a / b)
