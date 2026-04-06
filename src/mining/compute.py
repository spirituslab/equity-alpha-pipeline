"""Compute all candidate signals with pivot field caching."""

import pandas as pd
from tqdm import tqdm

from src.mining.enumeration import CandidateSpec
from src.mining.transforms import apply_transform


def compute_all_candidates(
    panel,
    specs: list[CandidateSpec],
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute all candidate signals, caching pivoted fields.

    Returns dict mapping candidate name -> (date x gvkey) DataFrame.
    Silently skips candidates that fail (missing fields, etc.).
    """
    pivot_cache = {}
    results = {}
    failed = 0

    iterator = tqdm(specs, desc="Computing candidates") if show_progress else specs

    for spec in iterator:
        try:
            df = apply_transform(panel, spec, pivot_cache)
            n_valid = df.count().sum()
            if n_valid > 1000:  # skip nearly-empty signals
                results[spec.name] = df
            else:
                failed += 1
        except Exception:
            failed += 1

    if show_progress:
        print(f"  Computed {len(results)} candidates, {failed} failed/empty")

    return results
