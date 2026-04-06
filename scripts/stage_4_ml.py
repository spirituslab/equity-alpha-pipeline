"""Stage 4: ML Extension — compare Ridge, ElasticNet, XGBoost vs baseline.

For each ML method:
1. Generate composite alpha via walk-forward ML prediction
2. Run backtest (naive L/S for speed)
3. Compare IC, Sharpe, turnover vs EW and IC-weighted baselines
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
from src.signals.combine import combine_signals
from src.analytics.ic import compute_ic_series, ic_summary
from src.analytics.performance import sharpe_ratio
from src.analytics.attribution import factor_attribution, print_attribution
from src.analytics.bootstrap import block_bootstrap_sharpe
from src.portfolio.backtest import WalkForwardBacktest


def run_backtest_for_alpha(
    name: str,
    alpha: pd.DataFrame,
    config: PipelineConfig,
    returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
    rf: pd.Series,
    beta: pd.DataFrame,
    sectors: pd.DataFrame,
    universe: pd.DataFrame,
):
    """Run naive L/S backtest and report results for a given alpha."""
    bt = WalkForwardBacktest(
        config=config,
        alpha_scores=alpha,
        stock_returns=returns,
        factor_returns=ff_factors,
        beta=beta,
        sector_labels=sectors,
        universe=universe,
        rf=rf,
        use_optimizer=False,
    )
    result = bt.run(
        start_date=config.dates.burn_in_end,
        end_date=config.dates.end,
    )

    net = result.net_returns
    if len(net) == 0:
        print(f"  {name}: No returns!")
        return None

    # Compute IC of the composite alpha itself
    ic = compute_ic_series(alpha, returns, universe, end_date=config.dates.val_end)
    ic_stats = ic_summary(ic)

    # Full-sample metrics
    full_sharpe = sharpe_ratio(net)
    full_ret = net.mean() * 12

    # OOS metrics
    oos_start = pd.Period("2015-01", "M")
    oos_net = net[net.index >= oos_start]
    oos_sharpe = sharpe_ratio(oos_net) if len(oos_net) > 12 else np.nan

    # Turnover
    avg_turnover = result.turnover.mean()

    # Attribution (ff_factors already excludes RF)
    attr = factor_attribution(net, ff_factors, rf)
    alpha_ann = attr.get("alpha", np.nan)
    alpha_t = attr.get("alpha_t_stat", np.nan)

    boot = block_bootstrap_sharpe(oos_net, n_bootstrap=5000) if len(oos_net) > 12 else {}

    return {
        "name": name,
        "composite_ic": ic_stats.get("Mean IC", np.nan),
        "composite_icir": ic_stats.get("ICIR", np.nan),
        "full_sharpe": full_sharpe,
        "full_return": full_ret,
        "oos_sharpe": oos_sharpe,
        "oos_ci_lo": boot.get("ci_lower", np.nan),
        "oos_ci_hi": boot.get("ci_upper", np.nan),
        "avg_turnover": avg_turnover,
        "ff_alpha": alpha_ann,
        "ff_alpha_t": alpha_t,
        "n_months": len(net),
    }


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)

    print("=" * 60)
    print("  STAGE 4: ML EXTENSION")
    print("=" * 60)

    # Load cached data
    print("\n  Loading cached data...")
    returns = pd.read_parquet(config.cache_path("returns.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))
    universe = pd.read_parquet(config.cache_path("universe.parquet"))

    # Load neutralized signals
    signal_names = config.signals.active
    neutral_signals = {}
    for name in signal_names:
        neutral_signals[name] = pd.read_parquet(config.cache_path(f"neutral_{name}.parquet"))
    print(f"  Loaded {len(neutral_signals)} neutralized signals")

    # Load sectors and FF factors
    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    print("  Loading Fama-French factors...")
    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    # ---- Generate alpha for each method ----
    methods = {
        "EW Blend": "equal",
        "IC-Weighted": "ic_weighted",
        "Ridge": "ridge",
        "ElasticNet": "elastic_net",
    }

    # Check if xgboost is available
    try:
        import xgboost
        methods["XGBoost"] = "xgboost"
    except ImportError:
        print("  XGBoost not installed, skipping.")

    results = []

    for method_name, method_key in methods.items():
        print(f"\n{'='*60}")
        print(f"  Running: {method_name}")
        print(f"{'='*60}")

        # Generate composite alpha
        alpha = combine_signals(
            neutral_signals,
            method=method_key,
            returns=returns,
            universe=universe,
            lookback=config.backtest.lookback_signal,
            train_window=config.backtest.lookback_ml,
            purge_gap=config.backtest.purge_gap,
        )

        # Cache
        alpha.to_parquet(config.cache_path(f"alpha_{method_key}.parquet"))

        # Run backtest
        res = run_backtest_for_alpha(
            method_name, alpha, config, returns, ff_no_rf, rf,
            beta, sectors, universe,
        )
        if res:
            results.append(res)

    # ---- Comparison Table ----
    print(f"\n\n{'='*80}")
    print(f"  ML COMPARISON SUMMARY")
    print(f"{'='*80}")

    df = pd.DataFrame(results)
    df = df.set_index("name")

    print(f"\n{'':20s} {'IC':>8s} {'ICIR':>8s} {'Sharpe':>8s} {'OOS SR':>8s} {'Turnover':>10s} {'FF α':>8s} {'α t':>8s}")
    print(f"{'':20s} {'':>8s} {'':>8s} {'(full)':>8s} {'(2015+)':>8s} {'':>10s} {'(ann)':>8s} {'':>8s}")
    print("-" * 90)
    for idx, row in df.iterrows():
        print(f"  {idx:18s} {row['composite_ic']:>8.4f} {row['composite_icir']:>8.4f} "
              f"{row['full_sharpe']:>8.4f} {row['oos_sharpe']:>8.4f} "
              f"{row['avg_turnover']:>10.4f} {row['ff_alpha']:>8.4f} {row['ff_alpha_t']:>8.2f}")

    print(f"\n  Key Questions:")
    baseline_sharpe = df.loc["IC-Weighted", "oos_sharpe"] if "IC-Weighted" in df.index else df.iloc[0]["oos_sharpe"]
    for idx, row in df.iterrows():
        if idx in ("Ridge", "ElasticNet", "XGBoost"):
            delta = row["oos_sharpe"] - baseline_sharpe
            print(f"    {idx} vs IC-Weighted: OOS Sharpe delta = {delta:+.4f}, "
                  f"Turnover delta = {row['avg_turnover'] - df.loc['IC-Weighted', 'avg_turnover']:+.4f}")

    print(f"\n  Stage 4 complete.")


if __name__ == "__main__":
    main()
