"""Transform library for signal candidate generation."""

from enum import Enum

import numpy as np
import pandas as pd

from src.data.cleaner import remove_infinities


class TransformType(Enum):
    LEVEL = "level"
    RATIO_TO_MKTCAP = "ratio_to_mktcap"
    RATIO_TO_ASSETS = "ratio_to_assets"
    RATIO_TO_EQUITY = "ratio_to_equity"
    RATIO_TO_SALES = "ratio_to_sales"
    GROWTH_YOY = "growth_yoy"
    GROWTH_QOQ = "growth_qoq"
    ACCELERATION = "acceleration"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    HIGH_LOW_RANGE = "high_low_range"
    TWO_FIELD_RATIO = "two_field_ratio"
    ANALYST_REVISION = "analyst_revision"
    ANALYST_SUE = "analyst_sue"
    ANALYST_DISPERSION = "analyst_dispersion"
    # New transforms (v2)
    DIFFERENCE = "difference"                 # field_a - field_b
    DIFFERENCE_RATIO = "difference_ratio"     # (field_a - field_b) / field_c (via spec.window as field_c index)
    NEGATE = "negate"                         # -field
    MOMENTUM_SKIP = "momentum_skip"           # shift(skip).rolling(window).mean()


def apply_transform(panel, spec, pivot_cache: dict) -> pd.DataFrame:
    """Apply a transform to generate a candidate signal.

    Args:
        panel: DataPanel instance
        spec: CandidateSpec with transform type and field names
        pivot_cache: dict of field -> pivoted DataFrame (populated lazily)

    Returns:
        (date x gvkey) DataFrame of raw signal exposures
    """
    t = spec.transform

    if t == TransformType.LEVEL:
        return _get_field(panel, spec.field_a, pivot_cache)

    elif t == TransformType.RATIO_TO_MKTCAP:
        field = _get_field(panel, spec.field_a, pivot_cache)
        mktcap = _get_mktcap(panel, pivot_cache)
        return remove_infinities(field / mktcap)

    elif t == TransformType.RATIO_TO_ASSETS:
        field = _get_field(panel, spec.field_a, pivot_cache)
        atq = _get_field(panel, "atq", pivot_cache)
        return remove_infinities(field / atq)

    elif t == TransformType.RATIO_TO_EQUITY:
        field = _get_field(panel, spec.field_a, pivot_cache)
        ceqq = _get_field(panel, "ceqq", pivot_cache)
        return remove_infinities(field / ceqq)

    elif t == TransformType.RATIO_TO_SALES:
        field = _get_field(panel, spec.field_a, pivot_cache)
        saleq = _get_field(panel, "saleq", pivot_cache)
        return remove_infinities(field / saleq)

    elif t == TransformType.GROWTH_YOY:
        field = _get_field(panel, spec.field_a, pivot_cache)
        return remove_infinities(field / field.shift(12) - 1)

    elif t == TransformType.GROWTH_QOQ:
        field = _get_field(panel, spec.field_a, pivot_cache)
        return remove_infinities(field / field.shift(3) - 1)

    elif t == TransformType.ACCELERATION:
        field = _get_field(panel, spec.field_a, pivot_cache)
        growth = remove_infinities(field / field.shift(12) - 1)
        return growth - growth.shift(12)

    elif t == TransformType.MOMENTUM:
        trt1m = _get_field(panel, "trt1m", pivot_cache)
        window = spec.window or 12
        return trt1m.shift(1).rolling(window=window).mean()

    elif t == TransformType.VOLATILITY:
        trt1m = _get_field(panel, "trt1m", pivot_cache)
        window = spec.window or 12
        return -trt1m.rolling(window=window).std()  # negative: low vol = high signal

    elif t == TransformType.HIGH_LOW_RANGE:
        prchm = _get_field(panel, "prchm", pivot_cache)
        prclm = _get_field(panel, "prclm", pivot_cache)
        prccm = _get_field(panel, "prccm", pivot_cache)
        return remove_infinities((prchm - prclm) / prccm)

    elif t == TransformType.TWO_FIELD_RATIO:
        a = _get_field(panel, spec.field_a, pivot_cache)
        b = _get_field(panel, spec.field_b, pivot_cache)
        return remove_infinities(a / b)

    elif t == TransformType.ANALYST_REVISION:
        numup = _get_field(panel, "NUMUP", pivot_cache)
        numdown = _get_field(panel, "NUMDOWN", pivot_cache)
        numest = _get_field(panel, "NUMEST", pivot_cache)
        return remove_infinities((numup - numdown) / numest)

    elif t == TransformType.ANALYST_SUE:
        surpmean = _get_field(panel, "surpmean", pivot_cache)
        surpstdev = _get_field(panel, "surpstdev", pivot_cache)
        return remove_infinities(surpmean / surpstdev)

    elif t == TransformType.ANALYST_DISPERSION:
        surpstdev = _get_field(panel, "surpstdev", pivot_cache)
        fy1 = _get_field(panel, "FY_1", pivot_cache)
        return remove_infinities(-surpstdev / fy1.abs())  # negative: low dispersion = good

    elif t == TransformType.DIFFERENCE:
        a = _get_field(panel, spec.field_a, pivot_cache)
        b = _get_field(panel, spec.field_b, pivot_cache)
        return a - b

    elif t == TransformType.DIFFERENCE_RATIO:
        # (field_a - field_b) / field_c
        a = _get_field(panel, spec.field_a, pivot_cache)
        b = _get_field(panel, spec.field_b, pivot_cache)
        if spec.field_c == "__mktcap__":
            c = _get_mktcap(panel, pivot_cache)
        else:
            c = _get_field(panel, spec.field_c, pivot_cache)
        return remove_infinities((a - b) / c)

    elif t == TransformType.NEGATE:
        field = _get_field(panel, spec.field_a, pivot_cache)
        return -field

    elif t == TransformType.MOMENTUM_SKIP:
        trt1m = _get_field(panel, "trt1m", pivot_cache)
        skip = spec.sign  # reuse sign field for skip months
        window = spec.window or 11
        return trt1m.shift(skip).rolling(window=window).mean()

    else:
        raise ValueError(f"Unknown transform: {t}")


def _get_field(panel, field: str, cache: dict) -> pd.DataFrame:
    """Get pivoted field from cache or compute and cache."""
    if field not in cache:
        cache[field] = panel.pivot(field)
    return cache[field]


def _get_mktcap(panel, cache: dict) -> pd.DataFrame:
    """Get market cap (prccm * cshoq) from cache."""
    key = "__mktcap__"
    if key not in cache:
        prccm = _get_field(panel, "prccm", cache)
        cshoq = _get_field(panel, "cshoq", cache)
        cache[key] = prccm * cshoq
    return cache[key]
