"""Auto-generated signal: cheq_accel"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class CheqAccel(Factor):
    """YoY acceleration of cheq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.159, Hit Rate=0.562,
        Turnover=0.179, Spread t=2.84
    Validation ICIR (2005-2014): 0.134
    """

    name = "cheq_accel"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("cheq")
        growth = remove_infinities(field / field.shift(12) - 1)
        return growth - growth.shift(12)
