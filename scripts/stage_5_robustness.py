"""Stage 5: Robustness battery.

Tests:
1. Signal ablation (drop one signal at a time)
2. Subperiod stability (2015-2017 vs 2018-2019)
3. Cost sensitivity (10 bps vs 30 bps round-trip)
4. Neutralization sensitivity (none vs sector-only vs full)
5. Regime analysis (bull vs bear)
6. Turnover penalty sweep (kappa = 0 to 0.05)
7. Long vs short leg attribution
"""

import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.signals.zscore import standardize_signal
from src.signals.neutralize import neutralize_signal
from src.signals.combine import combine_signals
from src.analytics.performance import sharpe_ratio, max_drawdown
from src.analytics.attribution import factor_attribution
from src.analytics.bootstrap import block_bootstrap_sharpe
from src.analytics.ic import compute_ic_series, ic_summary, ic_decay_analysis
from src.portfolio.backtest import WalkForwardBacktest


def quick_backtest(alpha, config, returns, ff_no_rf, rf, beta, sectors, universe):
    """Run naive L/S backtest and return net returns Series."""
    bt = WalkForwardBacktest(
        config=config, alpha_scores=alpha, stock_returns=returns,
        factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
        universe=universe, rf=rf, use_optimizer=False,
    )
    result = bt.run(start_date=config.dates.burn_in_end, end_date=config.dates.end)
    return result


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)

    print("=" * 70)
    print("  STAGE 5: ROBUSTNESS BATTERY")
    print("=" * 70)

    # Load data
    returns = pd.read_parquet(config.cache_path("returns.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))
    universe = pd.read_parquet(config.cache_path("universe.parquet"))
    log_mcap = pd.read_parquet(config.cache_path("log_mcap.parquet"))

    signal_names = config.signals.active
    neutral_signals = {}
    z_signals = {}
    raw_signals = {}
    for name in signal_names:
        neutral_signals[name] = pd.read_parquet(config.cache_path(f"neutral_{name}.parquet"))
        z_signals[name] = pd.read_parquet(config.cache_path(f"zscore_{name}.parquet"))
        raw_signals[name] = pd.read_parquet(config.cache_path(f"raw_{name}.parquet"))

    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    # Baseline alpha
    baseline_alpha = pd.read_parquet(config.cache_path("composite_alpha.parquet"))
    baseline_result = quick_backtest(baseline_alpha, config, returns, ff_no_rf, rf, beta, sectors, universe)
    baseline_net = baseline_result.net_returns

    oos_start = pd.Period("2015-01", "M")
    oos_mid = pd.Period("2018-01", "M")

    # ================================================================
    # 1. SIGNAL ABLATION
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  1. SIGNAL ABLATION (drop one at a time)")
    print(f"{'='*70}")

    print(f"\n  {'Dropped':25s} {'Full SR':>8s} {'OOS SR':>8s} {'IC':>8s} {'Δ OOS SR':>10s}")
    print(f"  {'-'*65}")

    baseline_oos_sr = sharpe_ratio(baseline_net[baseline_net.index >= oos_start])
    print(f"  {'(none - baseline)':25s} {sharpe_ratio(baseline_net):>8.4f} {baseline_oos_sr:>8.4f} {'':>8s} {'':>10s}")

    for drop_name in signal_names:
        remaining = {n: s for n, s in neutral_signals.items() if n != drop_name}
        if len(remaining) == 0:
            continue
        alpha_ablated = combine_signals(remaining, method="ic_weighted",
                                         returns=returns, universe=universe,
                                         lookback=config.backtest.lookback_signal)
        result = quick_backtest(alpha_ablated, config, returns, ff_no_rf, rf, beta, sectors, universe)
        net = result.net_returns
        oos_net = net[net.index >= oos_start]
        full_sr = sharpe_ratio(net)
        oos_sr = sharpe_ratio(oos_net) if len(oos_net) > 12 else np.nan
        ic = compute_ic_series(alpha_ablated, returns, universe, end_date=config.dates.val_end)
        mean_ic = ic.mean() if len(ic) > 0 else np.nan
        delta = oos_sr - baseline_oos_sr
        marker = " <<<" if delta < -0.05 else ""
        print(f"  {drop_name:25s} {full_sr:>8.4f} {oos_sr:>8.4f} {mean_ic:>8.4f} {delta:>+10.4f}{marker}")

    # ================================================================
    # 2. SUBPERIOD STABILITY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  2. SUBPERIOD STABILITY")
    print(f"{'='*70}")

    periods = {
        "Development (1975-2004)": (pd.Period("1975-01", "M"), pd.Period("2004-12", "M")),
        "Validation (2005-2014)": (pd.Period("2005-01", "M"), pd.Period("2014-12", "M")),
        "OOS Early (2015-2017)": (oos_start, pd.Period("2017-12", "M")),
        "OOS Late (2018-2019)": (oos_mid, pd.Period("2019-12", "M")),
        "Full OOS (2015-2019)": (oos_start, pd.Period("2019-12", "M")),
    }

    print(f"\n  {'Period':30s} {'Sharpe':>8s} {'Return':>8s} {'Vol':>8s} {'MaxDD':>8s} {'Months':>8s}")
    print(f"  {'-'*72}")

    for pname, (pstart, pend) in periods.items():
        sub = baseline_net[(baseline_net.index >= pstart) & (baseline_net.index <= pend)]
        if len(sub) < 6:
            continue
        sr = sharpe_ratio(sub)
        ret = sub.mean() * 12
        vol = sub.std() * np.sqrt(12)
        mdd = max_drawdown(sub)
        print(f"  {pname:30s} {sr:>8.4f} {ret:>8.4f} {vol:>8.4f} {mdd:>8.4f} {len(sub):>8d}")

    # ================================================================
    # 3. COST SENSITIVITY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  3. COST SENSITIVITY")
    print(f"{'='*70}")

    cost_levels = [0, 5, 10, 15, 20, 30]
    print(f"\n  {'Cost (bps RT)':>15s} {'Full SR':>8s} {'OOS SR':>8s} {'Avg Cost/Mo':>12s}")
    print(f"  {'-'*50}")

    for cost_bps in cost_levels:
        cfg_copy = deepcopy(config)
        cfg_copy.costs.base_bps = cost_bps
        result = quick_backtest(baseline_alpha, cfg_copy, returns, ff_no_rf, rf, beta, sectors, universe)
        net = result.net_returns
        oos_net = net[net.index >= oos_start]
        avg_cost = result.costs.mean() * 10000 if len(result.costs) > 0 else 0
        print(f"  {cost_bps:>15d} {sharpe_ratio(net):>8.4f} {sharpe_ratio(oos_net):>8.4f} {avg_cost:>12.2f}")

    # ================================================================
    # 4. NEUTRALIZATION SENSITIVITY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  4. NEUTRALIZATION SENSITIVITY")
    print(f"{'='*70}")

    neut_configs = [
        ("None", False, False, False),
        ("Sector only", True, False, False),
        ("Size only", False, True, False),
        ("Beta only", False, False, True),
        ("Sector+Size", True, True, False),
        ("Full (S+S+B)", True, True, True),
    ]

    print(f"\n  {'Neutralization':20s} {'Full SR':>8s} {'OOS SR':>8s} {'FF α':>8s} {'α t':>8s}")
    print(f"  {'-'*55}")

    for nname, n_sec, n_size, n_beta in neut_configs:
        # Re-neutralize with different config
        neut_sigs = {}
        for sname in signal_names:
            neut_sigs[sname] = neutralize_signal(
                z_signals[sname],
                sector_labels=sectors if n_sec else pd.DataFrame(),
                log_mcap=log_mcap if n_size else pd.DataFrame(),
                beta=beta if n_beta else pd.DataFrame(),
                neutralize_sector=n_sec,
                neutralize_size=n_size,
                neutralize_beta=n_beta,
            )

        alpha_neut = combine_signals(neut_sigs, method="ic_weighted",
                                      returns=returns, universe=universe,
                                      lookback=config.backtest.lookback_signal)
        result = quick_backtest(alpha_neut, config, returns, ff_no_rf, rf, beta, sectors, universe)
        net = result.net_returns
        oos_net = net[net.index >= oos_start]
        attr = factor_attribution(net, ff_no_rf, rf)
        ff_alpha = attr.get("alpha", np.nan)
        ff_t = attr.get("alpha_t_stat", np.nan)
        print(f"  {nname:20s} {sharpe_ratio(net):>8.4f} {sharpe_ratio(oos_net):>8.4f} {ff_alpha:>8.4f} {ff_t:>8.2f}")

    # ================================================================
    # 5. REGIME ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  5. REGIME ANALYSIS (Bull vs Bear)")
    print(f"{'='*70}")

    # Classify months by trailing 12M S&P 500 return
    sp500_ret = panel.get_market_excess()
    sp500_trailing = sp500_ret.rolling(12).sum()

    bull_months = sp500_trailing[sp500_trailing > 0].index
    bear_months = sp500_trailing[sp500_trailing <= 0].index

    net_bull = baseline_net[baseline_net.index.isin(bull_months)]
    net_bear = baseline_net[baseline_net.index.isin(bear_months)]

    print(f"\n  {'Regime':15s} {'Sharpe':>8s} {'Return':>8s} {'Vol':>8s} {'Months':>8s}")
    print(f"  {'-'*45}")
    if len(net_bull) > 6:
        print(f"  {'Bull':15s} {sharpe_ratio(net_bull):>8.4f} {net_bull.mean()*12:>8.4f} {net_bull.std()*np.sqrt(12):>8.4f} {len(net_bull):>8d}")
    if len(net_bear) > 6:
        print(f"  {'Bear':15s} {sharpe_ratio(net_bear):>8.4f} {net_bear.mean()*12:>8.4f} {net_bear.std()*np.sqrt(12):>8.4f} {len(net_bear):>8d}")
    print(f"  {'All':15s} {sharpe_ratio(baseline_net):>8.4f} {baseline_net.mean()*12:>8.4f} {baseline_net.std()*np.sqrt(12):>8.4f} {len(baseline_net):>8d}")

    # ================================================================
    # 6. LONG vs SHORT LEG ATTRIBUTION
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  6. LONG vs SHORT LEG ATTRIBUTION")
    print(f"{'='*70}")

    long_rets = {}
    short_rets = {}
    for t, weights in baseline_result.holdings.items():
        t_plus_1 = t + 1
        if t_plus_1 not in returns.index:
            continue
        long_w = weights[weights > 0.001]
        short_w = weights[weights < -0.001]
        r = returns.loc[t_plus_1]
        if len(long_w) > 0:
            long_rets[t_plus_1] = (long_w * r.reindex(long_w.index).fillna(0)).sum() / long_w.sum()
        if len(short_w) > 0:
            short_rets[t_plus_1] = (short_w * r.reindex(short_w.index).fillna(0)).sum() / abs(short_w.sum())

    long_s = pd.Series(long_rets)
    short_s = pd.Series(short_rets)

    print(f"\n  {'Leg':15s} {'Sharpe':>8s} {'Return':>8s} {'Vol':>8s}")
    print(f"  {'-'*40}")
    if len(long_s) > 12:
        print(f"  {'Long':15s} {sharpe_ratio(long_s):>8.4f} {long_s.mean()*12:>8.4f} {long_s.std()*np.sqrt(12):>8.4f}")
    if len(short_s) > 12:
        print(f"  {'Short':15s} {sharpe_ratio(short_s):>8.4f} {short_s.mean()*12:>8.4f} {short_s.std()*np.sqrt(12):>8.4f}")

    # ================================================================
    # 7. BOOTSTRAP CONFIDENCE INTERVALS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  7. BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'='*70}")

    for pname, sub in [("Full sample", baseline_net),
                         ("OOS (2015-2019)", baseline_net[baseline_net.index >= oos_start])]:
        if len(sub) < 12:
            continue
        boot = block_bootstrap_sharpe(sub, n_bootstrap=10000, block_size=12)
        print(f"\n  {pname}:")
        print(f"    Sharpe: {boot['sharpe']:.4f}  CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]  SE: {boot['se']:.4f}")

    # ================================================================
    # 8. OOS IC DECAY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  8. OOS IC DECAY (signal persistence)")
    print(f"{'='*70}")

    print(f"\n  {'Signal':25s} {'Lag1':>7s} {'Lag3':>7s} {'Lag6':>7s} {'Lag12':>7s}")
    print(f"  {'-'*55}")

    for sname, sig in neutral_signals.items():
        decay = ic_decay_analysis(sig, returns, universe, max_lag=12, end_date=config.dates.end)
        if decay.empty:
            continue
        l1 = decay.loc[1, "Mean IC"] if 1 in decay.index else np.nan
        l3 = decay.loc[3, "Mean IC"] if 3 in decay.index else np.nan
        l6 = decay.loc[6, "Mean IC"] if 6 in decay.index else np.nan
        l12 = decay.loc[12, "Mean IC"] if 12 in decay.index else np.nan
        print(f"  {sname:25s} {l1:>7.4f} {l3:>7.4f} {l6:>7.4f} {l12:>7.4f}")

    print(f"\n{'='*70}")
    print(f"  Stage 5 complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
