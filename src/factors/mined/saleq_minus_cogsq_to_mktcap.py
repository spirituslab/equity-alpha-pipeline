"""Auto-generated signal: saleq_minus_cogsq_to_mktcap"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class SaleqMinusCogsqToMktcap(Factor):
    """(saleq - cogsq) / __mktcap__

    Mining discovery metrics (dev period 1975-2004):
        ICIR=0.358, Hit Rate=0.655,
        Turnover=0.051, Spread t=3.88
    Validation ICIR (2005-2014): 0.167
    """

    name = "saleq_minus_cogsq_to_mktcap"
    category = "value"

    def compute(self, panel) -> pd.DataFrame:
        a = panel.pivot("saleq")
        b = panel.pivot("cogsq")
        c = panel.pivot("prccm") * panel.pivot("cshoq")
        return remove_infinities((a - b) / c)
