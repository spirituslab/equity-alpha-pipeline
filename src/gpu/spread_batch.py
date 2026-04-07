"""GPU-batched decile spread computation.

Computes top-decile minus bottom-decile return spread for all signals
simultaneously, avoiding the Python date loop in report_card._compute_decile_spread.
"""

import numpy as np
import pandas as pd

from src.gpu.backend import GPU_AVAILABLE, to_cpu

if GPU_AVAILABLE:
    import cupy as cp


def batch_compute_decile_spread(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame = None,
    end_date: str = None,
    start_date: str = None,
    min_stocks: int = 50,
) -> dict[str, dict]:
    """GPU-batched decile spread for all signals.

    For each signal and date: sort stocks into deciles by signal value,
    compute mean return for top and bottom decile, spread = top - bottom.

    Returns dict[signal_name -> {"spread": float, "spread_t": float}]
    """
    if not GPU_AVAILABLE or len(signals) < 3:
        return _cpu_fallback(signals, returns, universe, end_date, start_date, min_stocks)

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

    stocks = all_stocks.intersection(returns.columns)
    if universe is not None and len(universe) > 0:
        stocks = stocks.intersection(universe.columns)

    dates = common_dates
    if end_date:
        dates = dates[dates <= pd.Period(end_date, "M")]
    if start_date:
        dates = dates[dates >= pd.Period(start_date, "M")]

    # Need t+1 returns
    ret_dates = set(returns.index)
    valid_dates = [d for d in dates if (d + 1) in ret_dates]
    dates = pd.PeriodIndex(valid_dates).sort_values()

    T = len(dates)
    N = len(stocks)

    if T < 12 or N < min_stocks:
        return {name: {"spread": np.nan, "spread_t": np.nan} for name in signal_names}

    print(f"    GPU Spread: {C} signals × {T} dates × {N} stocks")

    # Build tensors
    sig_np = np.full((C, T, N), np.nan, dtype=np.float32)
    for k, name in enumerate(signal_names):
        sig_np[k] = signals[name].reindex(index=dates, columns=stocks).values

    ret_dates_plus1 = pd.PeriodIndex([d + 1 for d in dates])
    ret_np = returns.reindex(index=ret_dates_plus1, columns=stocks).values.astype(np.float32)

    if universe is not None and len(universe) > 0:
        univ_np = universe.reindex(index=dates, columns=stocks).fillna(False).values.astype(bool)
    else:
        univ_np = np.ones((T, N), dtype=bool)

    sig_gpu = cp.asarray(sig_np)
    ret_gpu = cp.asarray(ret_np)
    univ_gpu = cp.asarray(univ_np)

    # Per signal: accumulate spread per date
    # spread_matrix: (C, T) — spread for each signal on each date
    spread_matrix = cp.full((C, T), cp.nan, dtype=cp.float32)

    for t in range(T):
        base_mask = univ_gpu[t] & ~cp.isnan(ret_gpu[t])  # (N,)
        base_idx = cp.where(base_mask)[0]
        n_base = len(base_idx)

        if n_base < min_stocks:
            continue

        r_vals = ret_gpu[t, base_idx]  # (n_base,)

        # Check per-signal validity
        sig_valid = ~cp.isnan(sig_gpu[:, t, :])  # (C, N)
        per_cand = sig_valid[:, base_idx]  # (C, n_base)
        valid_counts = per_cand.sum(axis=1)  # (C,)

        # Batch signals fully valid on base
        all_valid = valid_counts == n_base
        if cp.any(all_valid):
            av_idx = cp.where(all_valid)[0]
            s_batch = sig_gpu[av_idx, t, :][:, base_idx]  # (n_av, n_base)

            # Rank to get decile assignment
            ranks = cp.argsort(cp.argsort(s_batch, axis=1), axis=1)  # (n_av, n_base)
            deciles = (ranks.astype(cp.float32) / n_base * 10).astype(cp.int32)
            deciles = cp.clip(deciles, 0, 9)

            # Top decile (9) and bottom decile (0)
            top_mask = deciles == 9  # (n_av, n_base)
            bot_mask = deciles == 0

            # Mean return per decile
            r_expanded = r_vals[None, :].repeat(len(av_idx), axis=0)  # (n_av, n_base)

            top_sum = (r_expanded * top_mask).sum(axis=1)
            top_count = top_mask.sum(axis=1).astype(cp.float32)
            bot_sum = (r_expanded * bot_mask).sum(axis=1)
            bot_count = bot_mask.sum(axis=1).astype(cp.float32)

            top_mean = cp.where(top_count > 0, top_sum / top_count, cp.float32(0))
            bot_mean = cp.where(bot_count > 0, bot_sum / bot_count, cp.float32(0))

            spread_matrix[av_idx, t] = top_mean - bot_mean

        # Slow path for partially valid signals
        partial = (~all_valid) & (valid_counts >= min_stocks)
        for k_gpu in cp.where(partial)[0]:
            k = int(k_gpu)
            sig_mask = per_cand[k]
            s_idx = cp.where(sig_mask)[0]
            n_valid = len(s_idx)
            if n_valid < min_stocks:
                continue

            s_vals = sig_gpu[k, t, base_idx[s_idx]]
            r_sub = r_vals[s_idx]

            ranks = cp.argsort(cp.argsort(s_vals))
            deciles = (ranks.astype(cp.float32) / n_valid * 10).astype(cp.int32)
            deciles = cp.clip(deciles, 0, 9)

            top = deciles == 9
            bot = deciles == 0
            if top.sum() > 0 and bot.sum() > 0:
                spread_matrix[k, t] = r_sub[top].mean() - r_sub[bot].mean()

    # Move to CPU and compute summary statistics
    spread_np = to_cpu(spread_matrix)  # (C, T)

    result = {}
    for k, name in enumerate(signal_names):
        spreads = spread_np[k]
        valid = ~np.isnan(spreads)
        if valid.sum() < 12:
            result[name] = {"spread": np.nan, "spread_t": np.nan}
            continue

        s = spreads[valid]
        mean_spread = float(s.mean())
        std_spread = float(s.std())
        t_stat = mean_spread / (std_spread / np.sqrt(len(s))) if std_spread > 0 else 0
        result[name] = {"spread": mean_spread, "spread_t": t_stat}

    print(f"    GPU Spread complete: {C} signals")
    return result


def _cpu_fallback(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    end_date: str,
    start_date: str,
    min_stocks: int,
) -> dict[str, dict]:
    """CPU fallback for small batches."""
    result = {}
    for name, signal in signals.items():
        sig = signal.copy()
        if end_date:
            sig = sig[sig.index <= pd.Period(end_date, "M")]
        if start_date:
            sig = sig[sig.index >= pd.Period(start_date, "M")]

        spreads = []
        for t in sig.index:
            t1 = t + 1
            if t1 not in returns.index:
                continue
            s = sig.loc[t].dropna()
            r = returns.loc[t1].reindex(s.index).dropna()
            common = s.index.intersection(r.index)
            if len(common) < min_stocks:
                continue
            s_c = s.loc[common]
            r_c = r.loc[common]
            n = len(common)
            top = s_c.nlargest(n // 10).index
            bot = s_c.nsmallest(n // 10).index
            if len(top) > 0 and len(bot) > 0:
                spreads.append(r_c.loc[top].mean() - r_c.loc[bot].mean())

        if len(spreads) >= 12:
            s = np.array(spreads)
            mean_spread = float(s.mean())
            std_spread = float(s.std())
            t_stat = mean_spread / (std_spread / np.sqrt(len(s))) if std_spread > 0 else 0
            result[name] = {"spread": mean_spread, "spread_t": t_stat}
        else:
            result[name] = {"spread": np.nan, "spread_t": np.nan}

    return result
