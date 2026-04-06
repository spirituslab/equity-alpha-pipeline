"""Auto-generated signal: oancfy_div_rectq"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyDivRectq(Factor):
    """oancfy / rectq

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.271, Hit Rate=0.628,
        Turnover=0.092, Spread t=4.00
    Validation ICIR (2005-2014): 0.229
    """

    name = "oancfy_div_rectq"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("oancfy")
        b = panel.pivot("rectq")
        return remove_infinities(a / b)
