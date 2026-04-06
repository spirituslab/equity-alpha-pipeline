"""Stage 3: Walk-forward backtest.

Runs two backtest variants:
1. Naive L/S: top/bottom 50 stocks, equal-weight, dollar-neutral only
2. Constrained optimizer: dollar/beta/sector-neutral with turnover penalty

Uses pre-computed composite alpha from Stage 2.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.portfolio.backtest import WalkForwardBacktest, BacktestResult
from src.analytics.performance import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, compute_descriptive_stats
from src.analytics.attribution import factor_attribution, print_attribution
from src.analytics.bootstrap import block_bootstrap_sharpe


def print_backtest_summary(name: str, result: BacktestResult, ff_factors: pd.DataFrame, rf: pd.Series):
    """Print backtest diagnostics."""
    gross = result.gross_returns
    net = result.net_returns

    if len(gross) == 0:
        print(f"  {name}: No returns generated!")
        return

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {name}")
    print(f"{'='*60}")

    print(f"\n  Period: {gross.index[0]} to {gross.index[-1]} ({len(gross)} months)")

    # Performance
    print(f"\n  Performance (annualized):")
    print(f"    Gross Return:    {gross.mean() * 12:>8.4f}")
    print(f"    Gross Vol:       {gross.std() * (12**0.5):>8.4f}")
    print(f"    Gross Sharpe:    {sharpe_ratio(gross):>8.4f}")
    print(f"    Net Return:      {net.mean() * 12:>8.4f}")
    print(f"    Net Sharpe:      {sharpe_ratio(net):>8.4f}")
    print(f"    Sortino:         {sortino_ratio(net):>8.4f}")
    print(f"    Calmar:          {calmar_ratio(net):>8.4f}")
    print(f"    Max Drawdown:    {max_drawdown(net):>8.4f}")
    print(f"    Win Rate:        {(net > 0).mean():>8.4f}")

    # Turnover & costs
    print(f"\n  Trading:")
    print(f"    Avg Turnover:    {result.turnover.mean():>8.4f}")
    print(f"    Avg Cost/Month:  {result.costs.mean()*10000:>8.2f} bps")
    print(f"    Total Cost Drag: {result.costs.sum()*10000:>8.0f} bps")

    # Neutrality
    if result.realized_beta is not None and len(result.realized_beta) > 0:
        print(f"\n  Neutrality:")
        print(f"    Avg |Beta|:      {result.realized_beta.abs().mean():>8.4f}")
        print(f"    Max |Beta|:      {result.realized_beta.abs().max():>8.4f}")

    # Positions
    if result.n_long is not None and len(result.n_long) > 0:
        print(f"\n  Positions:")
        print(f"    Avg Long:        {result.n_long.mean():>8.0f}")
        print(f"    Avg Short:       {result.n_short.mean():>8.0f}")

    # Bootstrap CI
    print(f"\n  Bootstrap 95% CI on Sharpe:")
    boot = block_bootstrap_sharpe(net, n_bootstrap=5000, block_size=12)
    print(f"    Sharpe:          {boot['sharpe']:>8.4f}")
    print(f"    CI:              [{boot['ci_lower']:>7.4f}, {boot['ci_upper']:>7.4f}]")
    print(f"    SE:              {boot['se']:>8.4f}")

    # Factor attribution
    print_attribution(factor_attribution(net, ff_factors, rf))


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)

    print("=" * 60)
    print("  STAGE 3: WALK-FORWARD BACKTEST")
    print("=" * 60)

    # Load cached data
    print("\n  Loading cached data...")
    alpha = pd.read_parquet(config.cache_path("composite_alpha.parquet"))
    returns = pd.read_parquet(config.cache_path("returns.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))
    universe = pd.read_parquet(config.cache_path("universe.parquet"))

    # Load sectors
    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    # Load French factors
    print("  Loading Fama-French factors...")
    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Returns shape: {returns.shape}")
    print(f"  FF factors shape: {ff_factors.shape}")

    # ---- Backtest 1: Naive L/S ----
    print("\n" + "=" * 60)
    print("  Running Naive Long/Short backtest...")
    print("=" * 60)

    bt_naive = WalkForwardBacktest(
        config=config,
        alpha_scores=alpha,
        stock_returns=returns,
        factor_returns=ff_no_rf,
        beta=beta,
        sector_labels=sectors,
        universe=universe,
        rf=rf,
        use_optimizer=False,
    )
    result_naive = bt_naive.run(
        start_date=config.dates.burn_in_end,
        end_date=config.dates.end,
    )

    print_backtest_summary("Naive L/S (EW, dollar-neutral)", result_naive, ff_no_rf, rf)

    # Save results
    result_naive.gross_returns.to_csv(config.cache_path("bt_naive_gross.csv"))
    result_naive.net_returns.to_csv(config.cache_path("bt_naive_net.csv"))

    # ---- Backtest 2: Constrained Optimizer ----
    print("\n" + "=" * 60)
    print("  Running Constrained Optimizer backtest...")
    print("=" * 60)

    bt_opt = WalkForwardBacktest(
        config=config,
        alpha_scores=alpha,
        stock_returns=returns,
        factor_returns=ff_no_rf,
        beta=beta,
        sector_labels=sectors,
        universe=universe,
        rf=rf,
        use_optimizer=True,
    )
    result_opt = bt_opt.run(
        start_date=config.dates.burn_in_end,
        end_date=config.dates.end,
    )

    print_backtest_summary("Constrained L/S (optimizer)", result_opt, ff_no_rf, rf)

    # Save results
    result_opt.gross_returns.to_csv(config.cache_path("bt_opt_gross.csv"))
    result_opt.net_returns.to_csv(config.cache_path("bt_opt_net.csv"))

    # ---- OOS Comparison ----
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE COMPARISON (2015-2019)")
    print("=" * 60)

    oos_start = pd.Period("2015-01", "M")
    for name, result in [("Naive L/S", result_naive), ("Constrained", result_opt)]:
        oos_net = result.net_returns[result.net_returns.index >= oos_start]
        if len(oos_net) > 0:
            print(f"\n  {name}:")
            print(f"    OOS Sharpe:  {sharpe_ratio(oos_net):>8.4f}")
            print(f"    OOS Return:  {oos_net.mean() * 12:>8.4f}")
            boot = block_bootstrap_sharpe(oos_net, n_bootstrap=5000)
            print(f"    OOS CI:      [{boot['ci_lower']:>7.4f}, {boot['ci_upper']:>7.4f}]")

    print("\n  Stage 3 complete.")


if __name__ == "__main__":
    main()
