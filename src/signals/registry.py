"""Signal registry with auto-discovery and config-driven active set."""

from src.factors.base import Factor
from src.factors.momentum import Momentum12_2, STReversal
from src.factors.fundamental import ROE, GrossProfitability, AccrualRatio, AssetGrowth, EarningsYield
from src.factors.risk import IdiosyncraticVol

# Mined signals (discovered by signal mining machine)
from src.factors.mined.ibcomq_to_mktcap import IbcomqToMktcap
from src.factors.mined.oancfy_div_rectq import OancfyDivRectq
from src.factors.mined.epsfxq_to_mktcap import EpsfxqToMktcap
from src.factors.mined.ibcomq_yoy import IbcomqYoy
from src.factors.mined.cheq_yoy import CheqYoy


# All known signals — auto-discovered by importing factor modules
_ALL_SIGNALS: dict[str, type[Factor]] = {
    "momentum_12_2": Momentum12_2,
    "st_reversal": STReversal,
    "roe": ROE,
    "asset_growth": AssetGrowth,
    "gross_profitability": GrossProfitability,
    "accrual_ratio": AccrualRatio,
    "earnings_yield": EarningsYield,
    "idio_vol": IdiosyncraticVol,
    # Mined signals
    "ibcomq_to_mktcap": IbcomqToMktcap,
    "oancfy_div_rectq": OancfyDivRectq,
    "epsfxq_to_mktcap": EpsfxqToMktcap,
    "ibcomq_yoy": IbcomqYoy,
    "cheq_yoy": CheqYoy,
}


class SignalRegistry:
    """Registry for signal discovery and selection.

    Usage:
        registry = SignalRegistry(active_names=config.signals.active)
        for signal in registry.get_active():
            raw = signal.compute(panel)
    """

    def __init__(self, active_names: list[str] | None = None):
        self._all = _ALL_SIGNALS.copy()
        self._active_names = active_names or list(self._all.keys())

    def register(self, signal_cls: type[Factor]) -> None:
        """Register a new signal class."""
        self._all[signal_cls.name] = signal_cls

    def get_all(self) -> list[Factor]:
        """Return instances of all registered signals."""
        return [cls() for cls in self._all.values()]

    def get_active(self) -> list[Factor]:
        """Return instances of signals in the active set (from config)."""
        active = []
        for name in self._active_names:
            if name in self._all:
                active.append(self._all[name]())
            else:
                raise ValueError(f"Signal '{name}' not found in registry. "
                                 f"Available: {list(self._all.keys())}")
        return active

    def get_by_name(self, name: str) -> Factor:
        """Return a single signal instance by name."""
        if name not in self._all:
            raise ValueError(f"Signal '{name}' not found. Available: {list(self._all.keys())}")
        return self._all[name]()

    @property
    def active_names(self) -> list[str]:
        return self._active_names

    @property
    def all_names(self) -> list[str]:
        return list(self._all.keys())
