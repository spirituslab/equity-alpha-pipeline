"""Build sector mapping for all S&P 500 constituents.

Uses SEC EDGAR submissions API to get SIC codes:
1. Fetch ticker -> CIK map from SEC bulk JSON
2. For each matched CIK, fetch SIC from submissions endpoint
3. Map SIC -> Ken French 12-industry classification

Output: data/raw/sector_mapping.csv (gvkey, tic, sic, sector)
"""

import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sectors import sic_to_french_12

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
USER_AGENT = "Andy Wang andywang@example.com"


def get_unique_sp500_companies() -> pd.DataFrame:
    """Extract unique S&P 500 companies from CRSP/Compustat."""
    df = pd.read_csv(
        DATA_DIR / "compustat_crsp.csv",
        usecols=["gvkey", "tic", "lpermno", "sp500"],
        dtype=str,
        low_memory=False,
    )
    sp500 = df[df["sp500"] == "1"]
    companies = sp500.sort_values("tic").drop_duplicates("gvkey", keep="last")
    return companies[["gvkey", "tic", "lpermno"]].reset_index(drop=True)


def fetch_ticker_to_cik() -> dict[str, str]:
    """Fetch ticker -> CIK mapping from SEC EDGAR bulk JSON."""
    url = "https://www.sec.gov/files/company_tickers.json"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    ticker_to_cik = {}
    for entry in data.values():
        ticker_to_cik[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return ticker_to_cik


def fetch_sic_from_submissions(cik: str) -> int | None:
    """Fetch SIC code from EDGAR submissions endpoint for a given CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            sic = data.get("sic")
            return int(sic) if sic else None
    except Exception:
        return None


def clean_ticker(ticker: str) -> str:
    """Clean CRSP ticker for EDGAR matching."""
    if not isinstance(ticker, str):
        return ""
    t = ticker.split(".")[0]  # Remove .1, .2 suffixes
    t = t.rstrip("Q")         # Remove bankruptcy Q suffix
    t = t.replace(" ", "")
    return t.upper()


def main():
    print("Building sector mapping for S&P 500 constituents...")
    companies = get_unique_sp500_companies()
    print(f"Found {len(companies)} unique S&P 500 companies")

    # Step 1: Get ticker -> CIK map
    print("\nFetching ticker -> CIK mapping from SEC...")
    ticker_to_cik = fetch_ticker_to_cik()
    print(f"  Got {len(ticker_to_cik)} ticker -> CIK mappings")

    # Step 2: Match our tickers to CIKs
    matched = []
    unmatched = []
    for _, row in companies.iterrows():
        clean = clean_ticker(row["tic"])
        cik = ticker_to_cik.get(clean)
        if cik:
            matched.append({"gvkey": row["gvkey"], "tic": row["tic"], "clean_tic": clean, "cik": cik})
        else:
            unmatched.append(row)

    print(f"  Matched {len(matched)} tickers to CIKs, {len(unmatched)} unmatched")

    # Step 3: Fetch SIC codes via submissions API
    # SEC rate limit: 10 requests/sec
    print(f"\nFetching SIC codes from EDGAR submissions API...")
    results = []
    batch_size = 9  # stay under 10/sec

    for i, entry in enumerate(matched):
        if i % 100 == 0:
            print(f"  [{i}/{len(matched)}] {entry['clean_tic']}...")

        sic = fetch_sic_from_submissions(entry["cik"])
        if sic:
            results.append({
                "gvkey": entry["gvkey"],
                "tic": entry["tic"],
                "sic": sic,
                "sector": sic_to_french_12(sic),
            })

        # Rate limiting: sleep every batch_size requests
        if (i + 1) % batch_size == 0:
            time.sleep(1.0)

    print(f"  Got SIC codes for {len(results)} companies")

    # Step 4: Fill missing with "Other"
    result_df = pd.DataFrame(results)
    all_gvkeys = set(companies["gvkey"])
    mapped_gvkeys = set(result_df["gvkey"]) if len(result_df) > 0 else set()
    still_missing = all_gvkeys - mapped_gvkeys

    if still_missing:
        print(f"\n{len(still_missing)} companies without sector data → 'Other'")
        missing_rows = companies[companies["gvkey"].isin(still_missing)]
        missing_df = pd.DataFrame({
            "gvkey": missing_rows["gvkey"].values,
            "tic": missing_rows["tic"].values,
            "sic": 0,
            "sector": "Other",
        })
        result_df = pd.concat([result_df, missing_df], ignore_index=True)

    # Save
    output_path = DATA_DIR / "sector_mapping.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved sector mapping to {output_path}")
    print(f"\nSector distribution:")
    print(result_df["sector"].value_counts().to_string())
    print(f"\nTotal: {len(result_df)}")
    coverage = len(result_df[result_df["sector"] != "Other"]) / len(result_df) * 100
    print(f"Coverage (non-Other): {coverage:.1f}%")


if __name__ == "__main__":
    main()
