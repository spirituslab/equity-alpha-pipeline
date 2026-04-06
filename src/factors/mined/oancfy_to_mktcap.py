"""Auto-generated signal: oancfy_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class OancfyToMktcap(Factor):
    """oancfy / market_cap

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.414, Hit Rate=0.667,
        Turnover=0.145, Spread t=5.92
    Validation ICIR (2005-2014): 0.245
    """

    name = "oancfy_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        field = panel.pivot("oancfy")
        mktcap = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities(field / mktcap)
