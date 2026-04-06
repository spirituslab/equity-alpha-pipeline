"""Correlation-based deduplication of candidate signals."""

import numpy as np
import pandas as pd


def deduplicate(
    survivors: dict[str, object],
    signals: dict[str, pd.DataFrame],
    existing_signals: dict[str, pd.DataFrame],
    max_corr: float = 0.70,
    n_sample_dates: int = 60,
) -> list[str]:
    """Greedy correlation-based deduplication.

    1. Sort candidates by |dev ICIR| descending
    2. Existing signals are always in the accepted set
    3. For each new candidate, check correlation with all accepted
    4. Accept only if max correlation < threshold

    Returns list of accepted new signal names.
    """
    # Sort by absolute ICIR descending
    ranked = sorted(survivors.keys(), key=lambda n: abs(survivors[n].dev_icir), reverse=True)

    # Build accepted set starting with existing signals
    accepted_names = list(existing_signals.keys())
    accepted_signals = dict(existing_signals)

    for name in ranked:
        if name not in signals:
            continue
        sig = signals[name]

        # Check correlation with all accepted
        max_c = 0.0
        for acc_name, acc_sig in accepted_signals.items():
            c = _avg_cross_sectional_corr(sig, acc_sig, n_sample_dates)
            max_c = max(max_c, abs(c))

        if max_c < max_corr:
            accepted_names.append(name)
            accepted_signals[name] = sig
            survivors[name].passed_dedup = True

    # Return only new names (not existing)
    return [n for n in accepted_names if n not in existing_signals]


def _avg_cross_sectional_corr(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    n_samples: int = 60,
) -> float:
    """Average cross-sectional correlation between two signals."""
    common_dates = sig_a.index.intersection(sig_b.index)
    common_stocks = sig_a.columns.intersection(sig_b.columns)

    if len(common_dates) == 0 or len(common_stocks) < 20:
        return 0.0

    dates = common_dates
    if len(dates) > n_samples:
        rng = np.random.default_rng(42)
        dates = rng.choice(dates, size=n_samples, replace=False)

    corrs = []
    for t in dates:
        a = sig_a.loc[t, common_stocks].dropna()
        b = sig_b.loc[t, common_stocks].reindex(a.index).dropna()
        common = a.index.intersection(b.index)
        if len(common) > 20:
            corrs.append(np.corrcoef(a.loc[common], b.loc[common])[0, 1])

    return np.mean(corrs) if corrs else 0.0
