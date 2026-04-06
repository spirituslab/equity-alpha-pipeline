from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DateConfig:
    start: str = "1962-01"
    end: str = "2019-12"
    burn_in_end: str = "1974-12"
    dev_end: str = "2004-12"
    val_end: str = "2014-12"


@dataclass
class DataConfig:
    compustat_file: str = "compustat_crsp.csv"
    sp500_file: str = "sp500_returns.csv"
    sector_file: str = "sector_mapping.csv"
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"


@dataclass
class UniverseConfig:
    min_mcap_mm: float = 100
    min_history_months: int = 12
    sp500_only: bool = True


@dataclass
class SignalConfig:
    active: list[str] = field(default_factory=lambda: [
        "momentum_12_2", "st_reversal", "roe",
        "asset_growth", "gross_profitability", "accrual_ratio",
    ])
    winsorize_pct: float = 0.01
    neutralize_sector: bool = True
    neutralize_size: bool = True
    neutralize_beta: bool = True
    combination_method: str = "ic_weighted"


@dataclass
class BacktestConfig:
    lookback_cov: int = 60
    lookback_signal: int = 36
    lookback_ml: int = 60
    purge_gap: int = 2
    rebalance_freq: str = "monthly"
    n_long: int = 50
    n_short: int = 50


@dataclass
class OptConfig:
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.005
    max_stock_weight: float = 0.02
    max_sector_net: float = 0.05
    max_gross_leverage: float = 2.0
    max_beta_exposure: float = 0.05


@dataclass
class CostConfig:
    base_bps: float = 10
    stress_bps: float = 30


@dataclass
class PipelineConfig:
    dates: DateConfig = field(default_factory=DateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptConfig = field(default_factory=OptConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    output_dir: str = "outputs"

    _project_root: Path = field(default=None, repr=False)

    def __post_init__(self):
        if self._project_root is None:
            self._project_root = Path(__file__).parent.parent

    @property
    def project_root(self) -> Path:
        return self._project_root

    def raw_path(self, filename: str) -> Path:
        return self.project_root / self.data.raw_dir / filename

    def cache_path(self, filename: str) -> Path:
        path = self.project_root / self.data.cache_dir
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        project_root = path.parent.parent  # config/ -> project root

        return cls(
            dates=DateConfig(**raw.get("dates", {})),
            data=DataConfig(**raw.get("data", {})),
            universe=UniverseConfig(**raw.get("universe", {})),
            signals=SignalConfig(**raw.get("signals", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            optimization=OptConfig(**raw.get("optimization", {})),
            costs=CostConfig(**raw.get("costs", {})),
            output_dir=raw.get("output", {}).get("dir", "outputs"),
            _project_root=project_root,
        )
