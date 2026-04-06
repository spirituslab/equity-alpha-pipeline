"""GPU-batched Information Coefficient computation — v2.

Eliminates the Python date loop entirely by grouping dates by valid stock count
and processing each group as one massive GPU batch operation.

v0: 385,000 kernel calls (loop dates × candidates) → 35 min
v1: 500 kernel calls (loop dates, batch candidates) → 25 min
v2: ~10-20 kernel calls (group dates, batch everything) → ~1-2 min
"""

import numpy as np
import pandas as pd

from src.gpu.backend import GPU_AVAILABLE, to_cpu

if GPU_AVAILABLE:
    import cupy as cp


def batch_compute_ic(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    end_date: str = None,
) -> dict[str, pd.Series]:
    """Compute rank IC for all signals with minimal Python loops.

    Groups dates by valid stock count, processes each group in one GPU call.
    Falls back to CPU for small batches or when GPU is unavailable.
    """
    if not GPU_AVAILABLE or len(signals) < 5:
        from src.analytics.ic import compute_ic_series
        return {name: compute_ic_series(sig, returns, universe, end_date=end_date)
                for name, sig in signals.items()}

    # ---- Align dates; use UNION of stocks (not intersection) to match CPU behavior ----
    signal_names = sorted(signals.keys())
    common_dates = None
    all_stocks = None
    for sig in signals.values():
        if common_dates is None:
            common_dates = sig.index
            all_stocks = sig.columns
        else:
            common_dates = common_dates.intersection(sig.index)
            all_stocks = all_stocks.union(sig.columns)

    # Intersect with returns and universe (these define the valid universe)
    common_stocks = all_stocks.intersection(returns.columns)
    if len(universe) > 0:
        common_stocks = common_stocks.intersection(universe.columns)

    if end_date:
        common_dates = common_dates[common_dates <= pd.Period(end_date, "M")]

    ret_dates = set(returns.index)
    valid_dates = [d for d in common_dates if (d + 1) in ret_dates]
    dates = pd.PeriodIndex(valid_dates)
    stocks = common_stocks

    C = len(signal_names)
    T = len(dates)
    N = len(stocks)

    if T == 0 or N == 0:
        from src.analytics.ic import compute_ic_series
        return {name: compute_ic_series(sig, returns, universe, end_date=end_date)
                for name, sig in signals.items()}

    print(f"    GPU IC v2: {C} signals × {T} dates × {N} stocks")

    # ---- Build tensors on CPU, move to GPU ----
    sig_np = np.full((C, T, N), np.nan, dtype=np.float32)
    for k, name in enumerate(signal_names):
        sig_np[k] = signals[name].reindex(index=dates, columns=stocks).values

    ret_dates_plus1 = pd.PeriodIndex([d + 1 for d in dates])
    ret_np = returns.reindex(index=ret_dates_plus1, columns=stocks).values.astype(np.float32)

    if len(universe) > 0:
        univ_np = universe.reindex(index=dates, columns=stocks).fillna(False).values.astype(bool)
    else:
        univ_np = np.ones((T, N), dtype=bool)

    sig_gpu = cp.asarray(sig_np)    # (C, T, N)
    ret_gpu = cp.asarray(ret_np)    # (T, N)
    univ_gpu = cp.asarray(univ_np)  # (T, N)

    ic_matrix = cp.full((C, T), cp.nan, dtype=cp.float32)

    # ---- Precompute per-date valid masks ----
    base_masks = (~cp.isnan(ret_gpu)) & univ_gpu  # (T, N)
    base_counts = base_masks.sum(axis=1)           # (T,)

    # Per-candidate validity on base stocks: (C, T)
    sig_valid = ~cp.isnan(sig_gpu)                         # (C, T, N)
    per_cand_on_base = (sig_valid & base_masks[None, :, :]).sum(axis=2)  # (C, T)

    # Dates where ALL candidates are valid on ALL base stocks
    all_uniform = (per_cand_on_base == base_counts[None, :]).all(axis=0)  # (T,)

    uniform_t = cp.where(all_uniform & (base_counts >= 20))[0]
    residual_t = cp.where(~all_uniform & (base_counts >= 20))[0]

    n_uniform = int(len(uniform_t))
    n_residual = int(len(residual_t))
    print(f"    Uniform dates: {n_uniform}, Residual dates: {n_residual}")

    # ---- FAST PATH: Group uniform dates by valid count, batch everything ----
    if n_uniform > 0:
        uniform_counts = base_counts[uniform_t]  # (n_uniform,)
        unique_counts = cp.unique(uniform_counts)

        for uc in unique_counts:
            n_valid = int(uc)
            group_mask = uniform_counts == uc
            group_dates = uniform_t[group_mask]
            n_dates = int(len(group_dates))

            if n_dates == 0:
                continue

            # Extract valid stock indices for first date in group (all same since uniform)
            first_t = int(group_dates[0])
            valid_idx = cp.where(base_masks[first_t])[0][:n_valid]

            # Extract signal values: (C, n_dates, n_valid)
            s_group = sig_gpu[:, group_dates, :][:, :, valid_idx]  # (C, n_dates, n_valid)
            r_group = ret_gpu[group_dates, :][:, valid_idx]         # (n_dates, n_valid)

            # Flatten signals: (C * n_dates, n_valid)
            s_flat = s_group.reshape(C * n_dates, n_valid)

            # Rank signals: one argsort call for all C * n_dates vectors
            s_ranks_flat = cp.argsort(cp.argsort(s_flat, axis=1), axis=1).astype(cp.float32)

            # Rank returns: (n_dates, n_valid)
            r_ranks = cp.argsort(cp.argsort(r_group, axis=1), axis=1).astype(cp.float32)

            # Reshape signal ranks back: (C, n_dates, n_valid)
            s_ranks = s_ranks_flat.reshape(C, n_dates, n_valid)

            # Center
            s_centered = s_ranks - s_ranks.mean(axis=2, keepdims=True)  # (C, n_dates, n_valid)
            r_centered = r_ranks - r_ranks.mean(axis=1, keepdims=True)  # (n_dates, n_valid)

            # Batched correlation: (C, n_dates)
            numer = (s_centered * r_centered[None, :, :]).sum(axis=2)   # (C, n_dates)
            s_ss = (s_centered ** 2).sum(axis=2)                         # (C, n_dates)
            r_ss = (r_centered ** 2).sum(axis=1)                         # (n_dates,)
            denom = cp.sqrt(s_ss * r_ss[None, :])                        # (C, n_dates)

            valid_denom = denom > 0
            ic_group = cp.where(valid_denom, numer / denom, cp.nan)      # (C, n_dates)

            # Write back to ic_matrix
            for i, gt in enumerate(group_dates):
                ic_matrix[:, gt] = ic_group[:, i]

    # ---- SLOW PATH: Residual dates (per-date loop, but only ~5% of dates) ----
    if n_residual > 0:
        for t_gpu in residual_t:
            t = int(t_gpu)
            base_idx = cp.where(base_masks[t])[0]

            if len(base_idx) < 20:
                continue

            r_vals = ret_gpu[t, base_idx]
            r_ranks = cp.argsort(cp.argsort(r_vals)).astype(cp.float32)
            r_centered = r_ranks - r_ranks.mean()
            r_ss = (r_centered ** 2).sum()

            if r_ss == 0:
                continue

            # Check per-candidate validity
            sig_valid_t = sig_valid[:, t, :][:, base_idx]  # (C, N_base)
            valid_counts = sig_valid_t.sum(axis=1)          # (C,)

            # Batch candidates with all valid
            all_valid = valid_counts == len(base_idx)
            if int(all_valid.sum()) > 0:
                av_idx = cp.where(all_valid)[0]
                s_batch = sig_gpu[av_idx, t, :][:, base_idx]
                s_ranks = cp.argsort(cp.argsort(s_batch, axis=1), axis=1).astype(cp.float32)
                s_centered = s_ranks - s_ranks.mean(axis=1, keepdims=True)
                numer = (s_centered * r_centered[None, :]).sum(axis=1)
                s_ss = (s_centered ** 2).sum(axis=1)
                denom = cp.sqrt(s_ss * r_ss)
                valid_denom = denom > 0
                ic_matrix[av_idx, t] = cp.where(valid_denom, numer / denom, cp.nan)

            # Per-candidate fallback for partial valid
            partial = (~all_valid) & (valid_counts >= 20)
            for k_gpu in cp.where(partial)[0]:
                k = int(k_gpu)
                s_mask = sig_valid_t[k]
                s_idx = cp.where(s_mask)[0]
                if len(s_idx) < 20:
                    continue
                s_vals = sig_gpu[k, t, base_idx[s_idx]]
                s_rk = cp.argsort(cp.argsort(s_vals)).astype(cp.float32)
                r_rk = cp.argsort(cp.argsort(r_vals[s_idx])).astype(cp.float32)
                s_c = s_rk - s_rk.mean()
                r_c = r_rk - r_rk.mean()
                d = cp.sqrt((s_c ** 2).sum() * (r_c ** 2).sum())
                if d > 0:
                    ic_matrix[k, t] = (s_c * r_c).sum() / d

    # ---- Move back to CPU ----
    ic_np = to_cpu(ic_matrix)  # (C, T)

    result = {}
    date_index = pd.PeriodIndex([d + 1 for d in dates])
    for k, name in enumerate(signal_names):
        ic_series = pd.Series(ic_np[k], index=date_index, dtype=float)
        result[name] = ic_series.dropna()

    print(f"    GPU IC v2 complete: {C} signals × {T} dates")
    return result
