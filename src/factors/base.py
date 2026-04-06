from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """Abstract base class for all equity factors / signals.

    Every factor must define:
        name: unique identifier (used in config, caching, registry)
        category: grouping for reporting (momentum, quality, value, growth, risk, sentiment)
        compute(): returns (date x gvkey) DataFrame of raw exposures

    The Signal protocol is identical — Factor and Signal are the same thing.
    A factor becomes a "signal" once it passes through z-scoring and neutralization.
    """

    name: str = ""
    category: str = ""

    @abstractmethod
    def compute(self, panel) -> pd.DataFrame:
        """Compute raw factor exposures.

        Returns (date x gvkey) DataFrame where each cell is the raw
        factor exposure for stock gvkey at date t.

        Must use only data available at each date (no look-ahead).
        """
        ...
