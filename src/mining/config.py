"""Mining configuration: field classifications, thresholds, transform specs."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Field classifications from CRSP/Compustat
INCOME_FIELDS = ["ibq", "ibcomq", "saleq", "cogsq", "oibdpq", "epsfxq"]
CASH_FLOW_FIELDS = ["oancfy", "capxy", "ivncfy", "dvy", "dvpsxq"]
BALANCE_SHEET_FIELDS = [
    "atq", "ceqq", "seqq", "ltq", "lctq", "dlcq", "dlttq",
    "cheq", "rectq", "invtq", "acoq", "apq", "pstkq", "icaptq", "mibq",
]
SHARES_FIELDS = ["cshoq"]
PRICE_FIELDS = ["prccm", "prchm", "prclm", "trt1m", "trfm", "cshom"]
ANALYST_FIELDS = ["NUMEST", "NUMUP", "NUMDOWN", "FY_1", "LTG", "actual", "surpmean", "surpstdev"]

ALL_FUNDAMENTAL_FIELDS = INCOME_FIELDS + CASH_FLOW_FIELDS + BALANCE_SHEET_FIELDS + SHARES_FIELDS

# Economically meaningful two-field ratio pairs
# (numerator_group, denominator_group, category)
RATIO_PAIR_RULES = [
    (INCOME_FIELDS, BALANCE_SHEET_FIELDS, "quality"),     # profitability / efficiency
    (INCOME_FIELDS, INCOME_FIELDS, "quality"),             # margin ratios
    (CASH_FLOW_FIELDS, INCOME_FIELDS, "quality"),          # cash quality
    (CASH_FLOW_FIELDS, BALANCE_SHEET_FIELDS, "quality"),   # cash efficiency
    (BALANCE_SHEET_FIELDS, BALANCE_SHEET_FIELDS, "value"), # leverage / composition
]

# Economically meaningful difference pairs: (a - b)
DIFFERENCE_PAIRS = [
    # Gross profit: revenue - costs
    (["saleq"], ["cogsq"], "quality"),
    # Net income quality: income - cash flow (accruals)
    (["ibq", "ibcomq"], ["oancfy"], "quality"),
    # Operating income variants
    (["saleq"], ["cogsq", "apq"], "quality"),
    (["oibdpq"], ["cogsq"], "quality"),
    # Net debt / liquidity
    (["dlttq", "ltq"], ["cheq"], "value"),
    (["dlcq"], ["cheq"], "value"),
    # Working capital
    (["rectq", "invtq"], ["apq", "lctq"], "quality"),
    # Retained earnings proxy
    (["ibq"], ["dvpsxq"], "quality"),
]


@dataclass
class MiningConfig:
    # Quality thresholds
    min_icir_dev: float = 0.15
    min_hit_rate: float = 0.52
    max_turnover: float = 0.60
    min_spread_t: float = 2.0
    min_icir_val: float = 0.10
    max_correlation: float = 0.70

    # Transform toggles
    enable_level: bool = True
    enable_ratio_to_mktcap: bool = True
    enable_ratio_to_assets: bool = True
    enable_ratio_to_equity: bool = True
    enable_ratio_to_sales: bool = True
    enable_growth_yoy: bool = True
    enable_growth_qoq: bool = True
    enable_acceleration: bool = True
    enable_momentum: bool = True
    enable_volatility: bool = True
    enable_high_low_range: bool = True
    enable_two_field_ratio: bool = True
    enable_difference: bool = True
    enable_difference_ratio: bool = True
    enable_negate: bool = True
    enable_momentum_skip: bool = True
    enable_analyst: bool = True

    momentum_windows: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    volatility_windows: list[int] = field(default_factory=lambda: [3, 6, 12, 24])

    # Output
    factor_dir: str = "src/factors/mined"
    results_csv: str = "outputs/mining_results.csv"
    report_txt: str = "outputs/mining_report.txt"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MiningConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        thresholds = raw.get("thresholds", {})
        transforms = raw.get("transforms", {})
        return cls(
            min_icir_dev=thresholds.get("min_icir_dev", 0.15),
            min_hit_rate=thresholds.get("min_hit_rate", 0.52),
            max_turnover=thresholds.get("max_turnover", 0.60),
            min_spread_t=thresholds.get("min_spread_t", 2.0),
            min_icir_val=thresholds.get("min_icir_val", 0.10),
            max_correlation=thresholds.get("max_correlation", 0.70),
            enable_level=transforms.get("enable_level", True),
            enable_ratio_to_mktcap=transforms.get("enable_ratio_to_mktcap", True),
            enable_ratio_to_assets=transforms.get("enable_ratio_to_assets", True),
            enable_ratio_to_equity=transforms.get("enable_ratio_to_equity", True),
            enable_ratio_to_sales=transforms.get("enable_ratio_to_sales", True),
            enable_growth_yoy=transforms.get("enable_growth_yoy", True),
            enable_growth_qoq=transforms.get("enable_growth_qoq", True),
            enable_acceleration=transforms.get("enable_acceleration", True),
            enable_momentum=transforms.get("enable_momentum", True),
            enable_volatility=transforms.get("enable_volatility", True),
            enable_high_low_range=transforms.get("enable_high_low_range", True),
            enable_two_field_ratio=transforms.get("enable_two_field_ratio", True),
            enable_analyst=transforms.get("enable_analyst", True),
            momentum_windows=raw.get("momentum_windows", [3, 6, 9, 12]),
            volatility_windows=raw.get("volatility_windows", [3, 6, 12, 24]),
            factor_dir=raw.get("output", {}).get("factor_dir", "src/factors/mined"),
            results_csv=raw.get("output", {}).get("results_csv", "outputs/mining_results.csv"),
            report_txt=raw.get("output", {}).get("report_txt", "outputs/mining_report.txt"),
        )
