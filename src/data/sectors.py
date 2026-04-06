"""Sector assignment for cross-sectional neutralization.

The CRSP/Compustat merged file does not contain SIC/GICS codes.
This module provides sector labels using multiple fallback strategies:

1. Primary: Static CSV mapping (gvkey -> sector) if provided
2. Fallback: Ken French 12-industry classification via SIC codes
3. Last resort: Ticker-based lookup from SEC EDGAR via edgartools
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Ken French 12-industry SIC code ranges
# Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
FRENCH_12_INDUSTRIES = {
    "Consumer NonDurables": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Consumer Durables": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manufacturing": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    "Energy": [(1200, 1399), (2900, 2999)],
    "Chemicals": [(2800, 2829), (2840, 2899)],
    "Business Equipment": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telecom": [(4800, 4899)],
    "Utilities": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Healthcare": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Finance": [(6000, 6999)],
    "Other": [],  # Everything else
}


def sic_to_french_12(sic: int) -> str:
    """Map a single SIC code to Ken French 12-industry classification."""
    if pd.isna(sic) or sic == 0:
        return "Other"
    sic = int(sic)
    for industry, ranges in FRENCH_12_INDUSTRIES.items():
        if industry == "Other":
            continue
        for low, high in ranges:
            if low <= sic <= high:
                return industry
    return "Other"


def load_sector_mapping(config) -> pd.DataFrame | None:
    """Try to load static sector mapping CSV.

    Expected format: gvkey, sector (one row per gvkey)
    Returns DataFrame with gvkey index and 'sector' column, or None if not found.
    """
    path = config.raw_path(config.data.sector_file)
    if path.exists():
        df = pd.read_csv(path, dtype={"gvkey": str})
        df["gvkey"] = df["gvkey"].str.strip()
        return df.set_index("gvkey")
    return None


def build_sector_labels_from_sic(sic_series: pd.Series) -> pd.Series:
    """Convert SIC codes to French 12-industry labels."""
    return sic_series.apply(sic_to_french_12)


def assign_sectors(panel, method: str = "static") -> pd.DataFrame:
    """Assign sector labels to each stock at each date.

    Returns (date x gvkey) DataFrame of sector labels (strings).

    Methods:
        "static": Use static CSV mapping (time-invariant)
        "sic": Use SIC codes from data (if available)
    """
    config = panel.config

    if method == "static":
        mapping = load_sector_mapping(config)
        if mapping is not None:
            # Broadcast static mapping across all dates
            returns = panel.get_returns()
            sectors = pd.DataFrame(
                index=returns.index,
                columns=returns.columns,
                dtype=str,
            )
            for gvkey in sectors.columns:
                if gvkey in mapping.index:
                    sectors[gvkey] = mapping.loc[gvkey, "sector"]
                else:
                    sectors[gvkey] = "Other"
            return sectors

    # Fallback: check if SIC codes exist in raw data
    if "sic" in panel.raw.columns:
        sic_pivot = panel.pivot("sic")
        return sic_pivot.applymap(lambda x: sic_to_french_12(x) if pd.notna(x) else "Other")

    raise ValueError(
        "No sector data available. Please provide one of:\n"
        "  1. Static mapping CSV at data/raw/sector_mapping.csv (columns: gvkey, sector)\n"
        "  2. SIC codes in the CRSP/Compustat dataset\n"
        "Run scripts/build_sector_mapping.py to generate the mapping."
    )


def build_sector_dummies(sectors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert sector labels to dummy variables for neutralization.

    Returns dict mapping each unique sector to a (date x gvkey) boolean DataFrame.
    Drops the most common sector for identification (avoids multicollinearity).
    """
    unique_sectors = sorted(set(sectors.values.flatten()) - {np.nan, "nan", ""})
    if not unique_sectors:
        return {}

    # Drop the most frequent sector (identification)
    sector_counts = pd.Series(sectors.values.flatten()).value_counts()
    drop_sector = sector_counts.index[0]

    dummies = {}
    for sector in unique_sectors:
        if sector == drop_sector:
            continue
        dummies[sector] = (sectors == sector).astype(float)

    return dummies
