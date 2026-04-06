"""Enumerate all candidate signal specifications."""

from dataclasses import dataclass

from src.mining.config import (
    MiningConfig, ALL_FUNDAMENTAL_FIELDS, INCOME_FIELDS, CASH_FLOW_FIELDS,
    BALANCE_SHEET_FIELDS, ANALYST_FIELDS, RATIO_PAIR_RULES, DIFFERENCE_PAIRS,
)
from src.mining.transforms import TransformType


@dataclass
class CandidateSpec:
    name: str
    category: str
    transform: TransformType
    field_a: str
    field_b: str | None = None
    field_c: str | None = None  # third field for DIFFERENCE_RATIO: (a - b) / c
    window: int | None = None
    sign: int = 1


def enumerate_candidates(config: MiningConfig) -> list[CandidateSpec]:
    """Generate all candidate signal specifications."""
    specs = []
    seen_names = set()

    def add(spec: CandidateSpec):
        if spec.name not in seen_names:
            seen_names.add(spec.name)
            specs.append(spec)

    # --- Single-field transforms on fundamentals ---
    for field in ALL_FUNDAMENTAL_FIELDS:
        cat = _field_category(field)

        if config.enable_level:
            add(CandidateSpec(f"{field}_level", cat, TransformType.LEVEL, field))

        if config.enable_ratio_to_mktcap and field != "cshoq":
            add(CandidateSpec(f"{field}_to_mktcap", "value", TransformType.RATIO_TO_MKTCAP, field))

        if config.enable_ratio_to_assets and field != "atq":
            add(CandidateSpec(f"{field}_to_assets", "quality", TransformType.RATIO_TO_ASSETS, field))

        if config.enable_ratio_to_equity and field not in ("ceqq", "seqq"):
            add(CandidateSpec(f"{field}_to_equity", "quality", TransformType.RATIO_TO_EQUITY, field))

        if config.enable_ratio_to_sales and field != "saleq":
            add(CandidateSpec(f"{field}_to_sales", "quality", TransformType.RATIO_TO_SALES, field))

        if config.enable_growth_yoy:
            add(CandidateSpec(f"{field}_yoy", "growth", TransformType.GROWTH_YOY, field))

        if config.enable_growth_qoq:
            add(CandidateSpec(f"{field}_qoq", "growth", TransformType.GROWTH_QOQ, field))

        if config.enable_acceleration:
            add(CandidateSpec(f"{field}_accel", "growth", TransformType.ACCELERATION, field))

    # --- Price transforms ---
    if config.enable_momentum:
        for w in config.momentum_windows:
            add(CandidateSpec(f"momentum_{w}m", "momentum", TransformType.MOMENTUM, "trt1m", window=w))

    if config.enable_volatility:
        for w in config.volatility_windows:
            add(CandidateSpec(f"volatility_{w}m", "momentum", TransformType.VOLATILITY, "trt1m", window=w))

    if config.enable_high_low_range:
        add(CandidateSpec("high_low_range", "momentum", TransformType.HIGH_LOW_RANGE, "prchm"))

    # --- Two-field ratios (economically meaningful pairs) ---
    if config.enable_two_field_ratio:
        for num_fields, den_fields, cat in RATIO_PAIR_RULES:
            for a in num_fields:
                for b in den_fields:
                    if a == b:
                        continue
                    name = f"{a}_div_{b}"
                    # Skip if already covered by normalization transforms
                    if name not in seen_names:
                        add(CandidateSpec(name, cat, TransformType.TWO_FIELD_RATIO, a, b))

    # --- Difference transforms: (field_a - field_b) ---
    if config.enable_difference:
        for a_fields, b_fields, cat in DIFFERENCE_PAIRS:
            for a in a_fields:
                for b in b_fields:
                    if a == b:
                        continue
                    add(CandidateSpec(f"{a}_minus_{b}", cat, TransformType.DIFFERENCE, a, b))

    # --- Difference Ratio transforms: (field_a - field_b) / field_c ---
    if config.enable_difference_ratio:
        denominators = ["atq", "ceqq", "saleq"]  # common normalizers
        for a_fields, b_fields, cat in DIFFERENCE_PAIRS:
            for a in a_fields:
                for b in b_fields:
                    if a == b:
                        continue
                    for c in denominators:
                        if c in (a, b):
                            continue
                        name = f"{a}_minus_{b}_div_{c}"
                        add(CandidateSpec(name, cat, TransformType.DIFFERENCE_RATIO, a, b, field_c=c))
                    # Also normalize by market cap
                    name = f"{a}_minus_{b}_to_mktcap"
                    add(CandidateSpec(name, "value", TransformType.DIFFERENCE_RATIO, a, b, field_c="__mktcap__"))

    # --- Negate transforms ---
    if config.enable_negate:
        # Negate return (reversal), growth signals, volatility
        add(CandidateSpec("neg_trt1m", "momentum", TransformType.NEGATE, "trt1m"))
        for field in ALL_FUNDAMENTAL_FIELDS:
            add(CandidateSpec(f"neg_{field}_yoy", "growth", TransformType.NEGATE, f"_growth_yoy_{field}"))
            # Only makes sense if growth is already computed — skip, negate raw fields instead
        # Negate level fields that might work inverted (high = bad)
        for field in ["atq", "ltq", "dlttq", "dlcq", "invtq"]:
            add(CandidateSpec(f"neg_{field}_level", "value", TransformType.NEGATE, field))

    # --- Momentum with skip month ---
    if config.enable_momentum_skip:
        for skip in [1, 2, 3]:
            for window in [3, 6, 9, 11]:
                add(CandidateSpec(
                    f"momentum_skip{skip}_{window}m", "momentum",
                    TransformType.MOMENTUM_SKIP, "trt1m", window=window, sign=skip,
                ))

    # --- Analyst transforms ---
    if config.enable_analyst:
        add(CandidateSpec("analyst_revision", "sentiment", TransformType.ANALYST_REVISION, "NUMUP"))
        add(CandidateSpec("analyst_sue", "sentiment", TransformType.ANALYST_SUE, "surpmean"))
        add(CandidateSpec("analyst_dispersion", "sentiment", TransformType.ANALYST_DISPERSION, "surpstdev"))

        # Growth on analyst fields
        for field in ["FY_1", "LTG"]:
            if config.enable_growth_yoy:
                add(CandidateSpec(f"{field}_yoy", "sentiment", TransformType.GROWTH_YOY, field))
            add(CandidateSpec(f"{field}_level", "sentiment", TransformType.LEVEL, field))

    return specs


def _field_category(field: str) -> str:
    """Assign default category based on field type."""
    if field in INCOME_FIELDS:
        return "quality"
    if field in CASH_FLOW_FIELDS:
        return "quality"
    if field in BALANCE_SHEET_FIELDS:
        return "value"
    return "other"
