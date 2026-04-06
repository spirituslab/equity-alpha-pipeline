"""Signal report card — per-signal diagnostic suite.

For any signal, automatically produce:
- Mean IC, ICIR, t-stat, hit rate
- IC decay at lags 1-12 (alpha half-life)
- Decile spread (top vs bottom decile return)
- Signal turnover (rank stability month-to-month)
- Correlation with every other signal
- Marginal IC: improvement to composite if this signal is added

This is the "should I include this signal?" decision tool.
"""

import numpy as np
import pandas as pd

from src.analytics.ic import compute_ic_series, ic_summary, ic_decay_analysis


def signal_report_card(
    signal_name: str,
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    other_signals: dict[str, pd.DataFrame] = None,
    end_date: str = None,
) -> dict:
    """Generate comprehensive report card for a single signal.

    Args:
        signal_name: name of the signal
        signal: (date x gvkey) standardized signal values
        returns: (date x gvkey) stock returns (decimal)
        universe: (date x gvkey) boolean mask
        other_signals: dict of other signals for correlation / marginal IC
        end_date: limit evaluation to dates up to this point

    Returns:
        dict with all diagnostic metrics
    """
    report = {"name": signal_name}

    # 1. IC analysis
    ic = compute_ic_series(signal, returns, universe, end_date=end_date)
    report["ic_series"] = ic
    report.update(ic_summary(ic))

    # 2. IC decay
    decay = ic_decay_analysis(signal, returns, universe, max_lag=12, end_date=end_date)
    report["ic_decay"] = decay

    # 3. Decile spread
    report["decile_spread"] = _compute_decile_spread(signal, returns, universe, end_date=end_date)

    # 4. Signal turnover
    report["signal_turnover"] = _compute_signal_turnover(signal)

    # 5. Correlation with other signals
    if other_signals:
        corr = _compute_signal_correlations(signal_name, signal, other_signals)
        report["correlations"] = corr

    # 6. Marginal IC (improvement if added to equal-weight blend of others)
    if other_signals:
        report["marginal_ic"] = _compute_marginal_ic(
            signal, other_signals, returns, universe, end_date=end_date
        )

    return report


def print_report_card(report: dict) -> None:
    """Pretty-print a signal report card."""
    print(f"\n{'='*60}")
    print(f"  SIGNAL REPORT CARD: {report['name']}")
    print(f"{'='*60}")

    # IC stats
    for key in ["Mean IC", "Std IC", "ICIR", "Hit Rate", "t-stat", "N Months"]:
        if key in report:
            val = report[key]
            if isinstance(val, float):
                print(f"  {key:20s}: {val:>8.4f}")
            else:
                print(f"  {key:20s}: {val:>8}")

    # IC decay
    if "ic_decay" in report and not report["ic_decay"].empty:
        print(f"\n  IC Decay:")
        for lag, row in report["ic_decay"].iterrows():
            print(f"    Lag {lag:2d}: IC={row['Mean IC']:>7.4f}  ICIR={row['ICIR']:>7.4f}")

    # Decile spread
    if "decile_spread" in report:
        ds = report["decile_spread"]
        print(f"\n  Decile Spread:")
        print(f"    Top decile return:    {ds.get('top_decile_ret', np.nan):>8.4f}")
        print(f"    Bottom decile return: {ds.get('bot_decile_ret', np.nan):>8.4f}")
        print(f"    Long-short spread:    {ds.get('spread', np.nan):>8.4f}")
        print(f"    Spread t-stat:        {ds.get('spread_t', np.nan):>8.4f}")

    # Signal turnover
    if "signal_turnover" in report:
        print(f"\n  Signal Turnover:        {report['signal_turnover']:>8.4f}")

    # Correlations
    if "correlations" in report:
        print(f"\n  Correlations with other signals:")
        for other, corr in sorted(report["correlations"].items()):
            print(f"    {other:25s}: {corr:>7.4f}")

    # Marginal IC
    if "marginal_ic" in report:
        mi = report["marginal_ic"]
        print(f"\n  Marginal IC:")
        print(f"    Blend IC without:     {mi.get('ic_without', np.nan):>8.4f}")
        print(f"    Blend IC with:        {mi.get('ic_with', np.nan):>8.4f}")
        print(f"    Marginal improvement: {mi.get('marginal', np.nan):>8.4f}")

    print(f"{'='*60}\n")


def _compute_decile_spread(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    end_date: str = None,
) -> dict:
    """Compute average return of top vs bottom decile sorted by signal."""
    top_rets = []
    bot_rets = []

    dates = signal.index
    if end_date:
        dates = dates[dates <= pd.Period(end_date, "M")]

    for t in dates:
        t_plus_1 = t + 1
        if t_plus_1 not in returns.index or t not in universe.index:
            continue

        members = universe.loc[t]
        members = members[members].index

        s = signal.loc[t].reindex(members).dropna()
        r = returns.loc[t_plus_1].reindex(members).dropna()

        common = np.intersect1d(s.index.values, r.index.values)
        if len(common) < 50:
            continue

        s = s.loc[common]
        r = r.loc[common]

        n_decile = max(len(s) // 10, 1)
        sorted_idx = s.sort_values().index
        bot_stocks = sorted_idx[:n_decile]
        top_stocks = sorted_idx[-n_decile:]

        top_rets.append(r.loc[top_stocks].mean())
        bot_rets.append(r.loc[bot_stocks].mean())

    if not top_rets:
        return {}

    top_arr = np.array(top_rets)
    bot_arr = np.array(bot_rets)
    spread = top_arr - bot_arr
    n = len(spread)
    spread_mean = spread.mean()
    spread_std = spread.std()
    spread_t = spread_mean / (spread_std / np.sqrt(n)) if spread_std > 0 else 0

    return {
        "top_decile_ret": top_arr.mean() * 12,   # annualized
        "bot_decile_ret": bot_arr.mean() * 12,
        "spread": spread_mean * 12,
        "spread_t": spread_t,
        "n_months": n,
    }


def _compute_signal_turnover(signal: pd.DataFrame) -> float:
    """Average rank correlation between consecutive months.

    High turnover (low rank correlation) means the signal is noisy
    and will generate high trading costs.
    """
    corrs = []
    for i in range(1, len(signal)):
        s_prev = signal.iloc[i - 1].dropna()
        s_curr = signal.iloc[i].dropna()
        common = np.intersect1d(s_prev.index.values, s_curr.index.values)
        if len(common) < 20:
            continue
        r_prev = s_prev.loc[common].values.argsort().argsort().astype(float)
        r_curr = s_curr.loc[common].values.argsort().argsort().astype(float)
        corrs.append(np.corrcoef(r_prev, r_curr)[0, 1])

    # Turnover = 1 - rank_autocorrelation
    return 1 - np.mean(corrs) if corrs else np.nan


def _compute_signal_correlations(
    name: str,
    signal: pd.DataFrame,
    other_signals: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Average cross-sectional correlation with each other signal."""
    correlations = {}
    sample_dates = signal.dropna(how="all").index
    if len(sample_dates) > 60:
        rng = np.random.default_rng(42)
        sample_dates = rng.choice(sample_dates, size=60, replace=False)

    for other_name, other_sig in other_signals.items():
        if other_name == name:
            continue
        corrs = []
        for t in sample_dates:
            if t not in other_sig.index:
                continue
            s1 = signal.loc[t].dropna()
            s2 = other_sig.loc[t].reindex(s1.index).dropna()
            common = s1.index.intersection(s2.index)
            if len(common) > 20:
                corrs.append(np.corrcoef(s1.loc[common], s2.loc[common])[0, 1])
        correlations[other_name] = np.mean(corrs) if corrs else np.nan

    return correlations


def _compute_marginal_ic(
    signal: pd.DataFrame,
    other_signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    end_date: str = None,
) -> dict:
    """Compute IC improvement when adding this signal to an equal-weight blend of others."""
    # Blend without this signal
    others_list = [s for n, s in other_signals.items()]
    if not others_list:
        return {}

    # Align
    common_dates = signal.index
    common_cols = signal.columns
    for s in others_list:
        common_dates = common_dates.intersection(s.index)
        common_cols = common_cols.intersection(s.columns)

    aligned_others = [s.loc[common_dates, common_cols] for s in others_list]
    aligned_signal = signal.loc[common_dates, common_cols]

    stacked_without = np.stack([s.values for s in aligned_others], axis=0)
    blend_without = pd.DataFrame(
        np.nanmean(stacked_without, axis=0),
        index=common_dates, columns=common_cols,
    )

    stacked_with = np.stack([s.values for s in aligned_others] + [aligned_signal.values], axis=0)
    blend_with = pd.DataFrame(
        np.nanmean(stacked_with, axis=0),
        index=common_dates, columns=common_cols,
    )

    ic_without = compute_ic_series(blend_without, returns, universe, end_date=end_date)
    ic_with = compute_ic_series(blend_with, returns, universe, end_date=end_date)

    return {
        "ic_without": ic_without.mean() if len(ic_without) > 0 else np.nan,
        "ic_with": ic_with.mean() if len(ic_with) > 0 else np.nan,
        "marginal": (ic_with.mean() - ic_without.mean()) if len(ic_with) > 0 and len(ic_without) > 0 else np.nan,
    }
