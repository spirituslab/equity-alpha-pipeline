"""Auto-generated signal: oancfy_level"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyLevel(Factor):
    """Raw oancfy

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.315, Hit Rate=0.647,
        Turnover=0.085, Spread t=2.36
    Validation ICIR (2005-2014): 0.119
    """

    name = "oancfy_level"
    category = "quality"

    def compute(self, panel) -> pd.DataFrame:
        return panel.pivot("oancfy")
