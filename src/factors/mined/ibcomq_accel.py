"""Auto-generated signal: ibcomq_accel"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class IbcomqAccel(Factor):
    """YoY acceleration of ibcomq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.279, Hit Rate=0.610,
        Turnover=0.262, Spread t=3.42
    Validation ICIR (2005-2014): 0.138
    """

    name = "ibcomq_accel"
    category = "growth"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("ibcomq")
        growth = remove_infinities(field / field.shift(12) - 1)
        return growth - growth.shift(12)
