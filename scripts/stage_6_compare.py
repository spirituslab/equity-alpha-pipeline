"""Stage 6: Compare expanded signal set (6 original + 5 mined) vs baseline (6 original).

Re-runs stages 1-3 with the expanded config, then compares against saved baseline results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.signals.registry import SignalRegistry
from src.signals.zscore import standardize_signal
from src.signals.neutralize import neutralize_all_signals, verify_neutralization
from src.signals.combine import combine_signals
from src.portfolio.backtest import WalkForwardBacktest
from src.analytics.performance import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
from src.analytics.attribution import factor_attribution, print_attribution
from src.analytics.bootstrap import block_bootstrap_sharpe
from src.analytics.ic import compute_ic_series, ic_summary


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)

    print("=" * 70)
    print("  STAGE 6: EXPANDED SIGNAL SET COMPARISON")
    print(f"  Active signals: {config.signals.active}")
    print("=" * 70)

    # ---- Stage 1: Compute new signals ----
    print("\n  Computing new mined signals...")
    registry = SignalRegistry(active_names=config.signals.active)

    for signal in registry.get_active():
        cache_file = config.cache_path(f"raw_{signal.name}.parquet")
        if cache_file.exists():
            print(f"    {signal.name}: cached")
        else:
            print(f"    {signal.name}: computing...")
            raw = signal.compute(panel)
            raw.to_parquet(cache_file)
            print(f"      shape={raw.shape}, non-null={raw.count().sum():,}")

    # Load control variables
    universe = pd.read_parquet(config.cache_path("universe.parquet"))
    log_mcap = pd.read_parquet(config.cache_path("log_mcap.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))
    returns = pd.read_parquet(config.cache_path("returns.parquet"))

    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    # ---- Stage 2: Standardize + neutralize + combine ----
    print("\n  Standardizing and neutralizing 11 signals...")
    z_signals = {}
    for signal in registry.get_active():
        raw = pd.read_parquet(config.cache_path(f"raw_{signal.name}.parquet"))
        z = standardize_signal(raw, winsorize_pct=config.signals.winsorize_pct)
        z_signals[signal.name] = z

    neutral_signals = neutralize_all_signals(
        z_signals,
        sector_labels=sectors,
        log_mcap=log_mcap,
        beta=beta,
        neutralize_sector=config.signals.neutralize_sector,
        neutralize_size=config.signals.neutralize_size,
        neutralize_beta=config.signals.neutralize_beta,
    )

    # Cache neutralized
    for name, sig in neutral_signals.items():
        sig.to_parquet(config.cache_path(f"neutral_{name}.parquet"))

    # Combine
    print(f"\n  Combining 11 signals (IC-weighted)...")
    composite_expanded = combine_signals(
        neutral_signals,
        method="ic_weighted",
        returns=returns,
        universe=universe,
        lookback=config.backtest.lookback_signal,
    )
    composite_expanded.to_parquet(config.cache_path("composite_alpha_expanded.parquet"))

    # Composite IC
    ic_expanded = compute_ic_series(composite_expanded, returns, universe, end_date=config.dates.val_end)
    ic_stats = ic_summary(ic_expanded)
    print(f"    Composite IC: {ic_stats.get('Mean IC', 0):.4f}, ICIR: {ic_stats.get('ICIR', 0):.4f}")

    # ---- Stage 3: Backtest ----
    print("\n  Running backtest with expanded signal set...")
    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    bt = WalkForwardBacktest(
        config=config,
        alpha_scores=composite_expanded,
        stock_returns=returns,
        factor_returns=ff_no_rf,
        beta=beta,
        sector_labels=sectors,
        universe=universe,
        rf=rf,
        use_optimizer=False,
    )
    result_expanded = bt.run(start_date=config.dates.burn_in_end, end_date=config.dates.end)

    # ---- Load baseline for comparison ----
    print("\n  Loading baseline results...")
    baseline_net = pd.read_csv(config.cache_path("bt_naive_net.csv"), index_col=0, parse_dates=False)
    baseline_net.index = pd.PeriodIndex(baseline_net.index, freq="M")
    baseline_net = baseline_net.iloc[:, 0]

    expanded_net = result_expanded.net_returns
    expanded_gross = result_expanded.gross_returns

    oos_start = pd.Period("2015-01", "M")
    baseline_oos = baseline_net[baseline_net.index >= oos_start]
    expanded_oos = expanded_net[expanded_net.index >= oos_start]

    # ---- Comparison Table ----
    print(f"\n{'='*70}")
    print(f"  COMPARISON: BASELINE (6 signals) vs EXPANDED (11 signals)")
    print(f"{'='*70}")

    print(f"\n  {'Metric':30s} {'Baseline (6)':>15s} {'Expanded (11)':>15s} {'Delta':>10s}")
    print(f"  {'-'*72}")

    metrics = [
        ("Full-Sample Sharpe (net)", sharpe_ratio(baseline_net), sharpe_ratio(expanded_net)),
        ("Full-Sample Return (ann)", baseline_net.mean() * 12, expanded_net.mean() * 12),
        ("Full-Sample Vol (ann)", baseline_net.std() * np.sqrt(12), expanded_net.std() * np.sqrt(12)),
        ("Max Drawdown", max_drawdown(baseline_net), max_drawdown(expanded_net)),
        ("OOS Sharpe (2015-2019)", sharpe_ratio(baseline_oos), sharpe_ratio(expanded_oos)),
        ("OOS Return (ann)", baseline_oos.mean() * 12, expanded_oos.mean() * 12),
        ("Avg Turnover", 2.19, result_expanded.turnover.mean()),
    ]

    for name, base_val, exp_val in metrics:
        delta = exp_val - base_val
        print(f"  {name:30s} {base_val:>15.4f} {exp_val:>15.4f} {delta:>+10.4f}")

    # Factor attribution comparison
    print(f"\n  Factor Attribution:")
    attr_base = factor_attribution(baseline_net, ff_no_rf, rf)
    attr_exp = factor_attribution(expanded_net, ff_no_rf, rf)

    print(f"    {'':20s} {'Baseline':>12s} {'Expanded':>12s}")
    print(f"    {'-'*45}")
    print(f"    {'FF Alpha (ann)':20s} {attr_base.get('alpha', 0):>12.4f} {attr_exp.get('alpha', 0):>12.4f}")
    print(f"    {'Alpha t-stat':20s} {attr_base.get('alpha_t_stat', 0):>12.2f} {attr_exp.get('alpha_t_stat', 0):>12.2f}")
    print(f"    {'R-squared':20s} {attr_base.get('r_squared', 0):>12.4f} {attr_exp.get('r_squared', 0):>12.4f}")

    # Bootstrap CIs
    print(f"\n  Bootstrap 95% CI on OOS Sharpe:")
    boot_base = block_bootstrap_sharpe(baseline_oos, n_bootstrap=10000)
    boot_exp = block_bootstrap_sharpe(expanded_oos, n_bootstrap=10000)
    print(f"    Baseline: {boot_base['sharpe']:.4f}  CI [{boot_base['ci_lower']:.4f}, {boot_base['ci_upper']:.4f}]")
    print(f"    Expanded: {boot_exp['sharpe']:.4f}  CI [{boot_exp['ci_lower']:.4f}, {boot_exp['ci_upper']:.4f}]")

    # Save expanded results
    expanded_net.to_csv(config.cache_path("bt_expanded_net.csv"))
    expanded_gross.to_csv(config.cache_path("bt_expanded_gross.csv"))

    print(f"\n{'='*70}")
    print(f"  Stage 6 complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
