"""GPU-batched signal turnover computation.

Computes rank autocorrelation (1 - turnover) for all signals simultaneously,
avoiding the Python date loop in report_card._compute_signal_turnover.

Same (C, T, N) tensor pattern as ic_batch.py.
"""

import numpy as np
import pandas as pd

from src.gpu.backend import GPU_AVAILABLE, to_cpu

if GPU_AVAILABLE:
    import cupy as cp


def batch_compute_turnover(
    signals: dict[str, pd.DataFrame],
    universe: pd.DataFrame = None,
    end_date: str = None,
    start_date: str = None,
    min_stocks: int = 20,
) -> dict[str, float]:
    """GPU-batched rank autocorrelation for all signals.

    Turnover = 1 - mean(rank_correlation(signal_t, signal_{t+1}))

    Falls back to CPU if GPU unavailable or batch is small.
    """
    if not GPU_AVAILABLE or len(signals) < 3:
        return _cpu_fallback(signals, universe, end_date, start_date, min_stocks)

    signal_names = sorted(signals.keys())
    C = len(signal_names)

    # Align dates and stocks
    common_dates = None
    all_stocks = None
    for sig in signals.values():
        if common_dates is None:
            common_dates = sig.index
            all_stocks = sig.columns
        else:
            common_dates = common_dates.intersection(sig.index)
            all_stocks = all_stocks.union(sig.columns)

    stocks = all_stocks
    if universe is not None and len(universe) > 0:
        stocks = stocks.intersection(universe.columns)

    dates = common_dates
    if end_date:
        dates = dates[dates <= pd.Period(end_date, "M")]
    if start_date:
        dates = dates[dates >= pd.Period(start_date, "M")]
    dates = dates.sort_values()

    T = len(dates)
    N = len(stocks)

    if T < 2 or N < min_stocks:
        return {name: np.nan for name in signal_names}

    print(f"    GPU Turnover: {C} signals × {T} dates × {N} stocks")

    # Build (C, T, N) tensor
    sig_np = np.full((C, T, N), np.nan, dtype=np.float32)
    for k, name in enumerate(signal_names):
        sig_np[k] = signals[name].reindex(index=dates, columns=stocks).values

    # Universe mask
    if universe is not None and len(universe) > 0:
        univ_np = universe.reindex(index=dates, columns=stocks).fillna(False).values.astype(bool)
    else:
        univ_np = np.ones((T, N), dtype=bool)

    sig_gpu = cp.asarray(sig_np)      # (C, T, N)
    univ_gpu = cp.asarray(univ_np)    # (T, N)

    # For each consecutive pair (t, t+1), compute rank correlation across all C signals
    # Accumulate sum of correlations and count per signal
    corr_sum = cp.zeros(C, dtype=cp.float64)
    corr_count = cp.zeros(C, dtype=cp.int32)

    for t in range(T - 1):
        # Valid stocks: in universe at both t and t+1, non-NaN in signal at both
        base_mask = univ_gpu[t] & univ_gpu[t + 1]  # (N,)

        # Per-signal validity: (C, N)
        valid_t = ~cp.isnan(sig_gpu[:, t, :])
        valid_t1 = ~cp.isnan(sig_gpu[:, t + 1, :])
        combined_valid = valid_t & valid_t1 & base_mask[None, :]  # (C, N)

        valid_counts = combined_valid.sum(axis=1)  # (C,)

        # Process signals with enough valid stocks
        enough = valid_counts >= min_stocks
        if not cp.any(enough):
            continue

        # For signals where all have the same valid set as base_mask,
        # we can batch them (fast path)
        base_valid_idx = cp.where(base_mask)[0]
        n_base = len(base_valid_idx)

        if n_base < min_stocks:
            continue

        # Check which signals are fully valid on base set
        all_valid_on_base = (combined_valid[:, base_valid_idx].sum(axis=1) == n_base)
        uniform_mask = enough & all_valid_on_base

        if cp.any(uniform_mask):
            u_idx = cp.where(uniform_mask)[0]
            # Extract: (n_uniform, n_base)
            s_t = sig_gpu[u_idx][:, t, :][:, base_valid_idx]
            s_t1 = sig_gpu[u_idx][:, t + 1, :][:, base_valid_idx]

            # Rank
            r_t = cp.argsort(cp.argsort(s_t, axis=1), axis=1).astype(cp.float32)
            r_t1 = cp.argsort(cp.argsort(s_t1, axis=1), axis=1).astype(cp.float32)

            # Center
            r_t -= r_t.mean(axis=1, keepdims=True)
            r_t1 -= r_t1.mean(axis=1, keepdims=True)

            # Correlation
            numer = (r_t * r_t1).sum(axis=1)
            denom = cp.sqrt((r_t ** 2).sum(axis=1) * (r_t1 ** 2).sum(axis=1))
            valid_denom = denom > 0
            corr = cp.where(valid_denom, numer / denom, cp.float32(0))

            corr_sum[u_idx] += corr.astype(cp.float64)
            corr_count[u_idx] += 1

        # Slow path: per-signal for non-uniform
        non_uniform = enough & ~uniform_mask
        for k_gpu in cp.where(non_uniform)[0]:
            k = int(k_gpu)
            valid_idx = cp.where(combined_valid[k])[0]
            if len(valid_idx) < min_stocks:
                continue
            s_t = sig_gpu[k, t, valid_idx]
            s_t1 = sig_gpu[k, t + 1, valid_idx]
            r_t = cp.argsort(cp.argsort(s_t)).astype(cp.float32)
            r_t1 = cp.argsort(cp.argsort(s_t1)).astype(cp.float32)
            r_t -= r_t.mean()
            r_t1 -= r_t1.mean()
            d = cp.sqrt((r_t ** 2).sum() * (r_t1 ** 2).sum())
            if d > 0:
                corr_sum[k] += float((r_t * r_t1).sum() / d)
                corr_count[k] += 1

    # Turnover = 1 - mean_rank_autocorrelation
    corr_sum_cpu = to_cpu(corr_sum)
    corr_count_cpu = to_cpu(corr_count)

    result = {}
    for k, name in enumerate(signal_names):
        if corr_count_cpu[k] > 0:
            mean_autocorr = corr_sum_cpu[k] / corr_count_cpu[k]
            result[name] = 1.0 - float(mean_autocorr)
        else:
            result[name] = np.nan

    print(f"    GPU Turnover complete: {C} signals")
    return result


def _cpu_fallback(
    signals: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    end_date: str,
    start_date: str,
    min_stocks: int,
) -> dict[str, float]:
    """CPU fallback using the same logic as report_card._compute_signal_turnover."""
    result = {}
    for name, signal in signals.items():
        sig = signal.copy()
        if end_date:
            sig = sig[sig.index <= pd.Period(end_date, "M")]
        if start_date:
            sig = sig[sig.index >= pd.Period(start_date, "M")]

        corrs = []
        for i in range(1, len(sig)):
            s_prev = sig.iloc[i - 1].dropna()
            s_curr = sig.iloc[i].dropna()
            common = np.intersect1d(s_prev.index.values, s_curr.index.values)
            if len(common) < min_stocks:
                continue
            r_prev = s_prev.loc[common].values.argsort().argsort().astype(float)
            r_curr = s_curr.loc[common].values.argsort().argsort().astype(float)
            corrs.append(np.corrcoef(r_prev, r_curr)[0, 1])

        result[name] = 1 - np.mean(corrs) if corrs else np.nan
    return result
