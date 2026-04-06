"""Auto-generated signal: epsfxq_level"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class EpsfxqLevel(Factor):
    """Raw epsfxq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.333, Hit Rate=0.664,
        Turnover=0.104, Spread t=3.45
    Validation ICIR (2005-2014): 0.206
    """

    name = "epsfxq_level"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        return panel.pivot("epsfxq")
