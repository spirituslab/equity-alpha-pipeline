"""Unified pipeline: mine → pre-select → stepwise → optimal signal set.

The complete workflow:
1. Enumerate ~770+ candidate signals (all transforms including DIFFERENCE, NEGATE)
2. Compute all candidates (with pivot caching)
3. Pre-select via IC evaluation (GPU-batched if available)
4. Validate survivors on holdout period
5. Deduplicate vs correlation
6. Forward stepwise portfolio selection (GPU-accelerated neutralization)
7. Output optimal signal set + final backtest comparison
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.gpu.backend import GPU_AVAILABLE
from src.mining.config import MiningConfig
from src.mining.runner import run_mining
from src.mining.stepwise import forward_stepwise_selection
from src.utils.logger import PipelineLogger


def main():
    pipeline_config = PipelineConfig.from_yaml("config/pipeline.yaml")
    mining_config = MiningConfig.from_yaml("config/mining.yaml")
    panel = DataPanel(pipeline_config)

    logger = PipelineLogger("full_pipeline")

    print("=" * 80)
    print("  FULL PIPELINE: MINE → PRE-SELECT → STEPWISE → OPTIMAL SET")
    print(f"  GPU: {'RTX 4080 SUPER (CuPy)' if GPU_AVAILABLE else 'Not available (CPU mode)'}")
    print("=" * 80)

    start_time = time.time()
    logger.step_start("Mining")

    # ---- Steps 1-7: Mining (enumerate, compute, evaluate, filter, dedup, codegen) ----
    try:
        run_mining(pipeline_config, mining_config)
        logger.step_complete("Mining", "Enumerated, evaluated, filtered, deduped")
    except Exception as e:
        logger.step_failed("Mining", str(e))
        raise

    # ---- Step 8: Load all survivors + original signals for stepwise ----
    print(f"\n{'='*80}")
    print("  STEP 8: FORWARD STEPWISE PORTFOLIO SELECTION")
    print(f"{'='*80}")

    # Load data
    returns = pd.read_parquet(pipeline_config.cache_path("returns.parquet"))
    universe = pd.read_parquet(pipeline_config.cache_path("universe.parquet"))
    log_mcap = pd.read_parquet(pipeline_config.cache_path("log_mcap.parquet"))
    beta = pd.read_parquet(pipeline_config.cache_path("beta.parquet"))

    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    # Load all candidates: original + mined survivors
    print("\n  Loading all candidate signals...")
    candidates = {}

    # Original 6
    for name in ["momentum_12_2", "st_reversal", "roe", "asset_growth",
                  "gross_profitability", "accrual_ratio"]:
        cache_file = pipeline_config.cache_path(f"raw_{name}.parquet")
        if cache_file.exists():
            candidates[name] = pd.read_parquet(cache_file)

    # Mined survivors (from codegen output)
    mined_dir = Path(pipeline_config.project_root) / mining_config.factor_dir
    for py_file in sorted(mined_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        signal_name = py_file.stem
        cache_file = pipeline_config.cache_path(f"raw_{signal_name}.parquet")
        if cache_file.exists():
            candidates[signal_name] = pd.read_parquet(cache_file)
        else:
            # Compute from mined factor class
            try:
                module_path = f"src.factors.mined.{signal_name}"
                mod = __import__(module_path, fromlist=[""])
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if isinstance(cls, type) and hasattr(cls, "name") and getattr(cls, "name", "") == signal_name:
                        raw = cls().compute(panel)
                        raw.to_parquet(cache_file)
                        candidates[signal_name] = raw
                        break
            except Exception:
                pass

    logger.info(f"Total candidates for stepwise: {len(candidates)}")
    logger.step_start("Stepwise Selection")

    # Build projection cache once
    from src.gpu.neutralize_batch import ProjectionCache
    print("\n  Building projection matrix cache...")
    proj_cache = ProjectionCache()
    proj_cache.build(
        sector_labels=sectors,
        log_mcap=log_mcap,
        beta=beta,
    )
    print(f"    Cached {len(proj_cache.projections)} dates")

    # Run stepwise
    history = forward_stepwise_selection(
        candidate_signals=candidates,
        returns=returns,
        universe=universe,
        sector_labels=sectors,
        log_mcap=log_mcap,
        beta=beta,
        factor_returns=ff_no_rf,
        rf=rf,
        config=pipeline_config,
        eval_metric="full_sharpe",
        max_signals=15,
        min_improvement=0.01,
        projection_cache=proj_cache,
    )

    # Save results
    if history:
        rows = [{
            "step": r.step,
            "added_signal": r.added_signal,
            "n_signals": r.n_signals,
            "full_sharpe": r.full_sharpe,
            "oos_sharpe": r.oos_sharpe,
            "ff_alpha": r.ff_alpha,
            "ff_alpha_t": r.ff_alpha_t,
            "improvement": r.improvement,
            "signal_set": "|".join(r.signal_set),
        } for r in history]
        df = pd.DataFrame(rows)
        df.to_csv(pipeline_config.cache_path("stepwise_v2_results.csv"), index=False)

    if history:
        best = max(history, key=lambda r: r.full_sharpe)
        logger.step_complete("Stepwise Selection",
                             f"{best.n_signals} signals, SR={best.full_sharpe:.4f}, OOS={best.oos_sharpe:.4f}")
        optimal_signals = best.signal_set
    else:
        logger.step_complete("Stepwise Selection", "No signals selected")
        optimal_signals = []

    # ================================================================
    # STEP 9: FINAL EVALUATION — Run optimal signals through full pipeline
    # Stepwise used equal-weight + naive L/S to select WHICH signals.
    # Now evaluate the final set with:
    #   - IC-weighted combination (adaptive signal weights)
    #   - Constrained optimizer (dollar/beta/sector neutral, turnover penalty)
    #   - Factor attribution (FF5+Mom)
    #   - Robustness: bootstrap CIs, long vs short leg
    # ================================================================
    if optimal_signals:
        print(f"\n{'='*80}")
        print(f"  STEP 9: FINAL EVALUATION — FULL PIPELINE ON OPTIMAL {len(optimal_signals)} SIGNALS")
        print(f"{'='*80}")
        logger.step_start("Final Evaluation")

        from src.signals.neutralize import neutralize_all_signals
        from src.signals.combine import combine_signals as combine_final
        from src.portfolio.backtest import WalkForwardBacktest
        from src.analytics.performance import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
        from src.analytics.attribution import factor_attribution, print_attribution
        from src.analytics.bootstrap import block_bootstrap_sharpe

        # Load raw signals for optimal set
        optimal_raw = {name: candidates[name] for name in optimal_signals if name in candidates}

        # Standardize + neutralize (full OLS, not projection matrix — for final numbers)
        from src.signals.zscore import standardize_signal as std_sig
        from src.signals.neutralize import neutralize_signal as neut_sig

        print(f"\n  Neutralizing {len(optimal_signals)} signals...")
        optimal_neutral = {}
        for name, raw in optimal_raw.items():
            z = std_sig(raw, winsorize_pct=0.01)
            n = neut_sig(z, sector_labels=sectors, log_mcap=log_mcap, beta=beta,
                         neutralize_sector=True, neutralize_size=True, neutralize_beta=True)
            optimal_neutral[name] = n

        # IC-weighted combination
        print(f"  Combining with IC-weighted method...")
        composite_ic = combine_final(
            optimal_neutral, method="ic_weighted",
            returns=returns, universe=universe,
            lookback=pipeline_config.backtest.lookback_signal,
        )

        # Also equal-weight for comparison
        composite_ew = combine_final(optimal_neutral, method="equal")

        oos_start = pd.Period("2015-01", "M")

        for combo_name, composite in [("Equal-Weight", composite_ew), ("IC-Weighted", composite_ic)]:
            print(f"\n  --- {combo_name} Combination ---")

            # Naive L/S backtest
            bt = WalkForwardBacktest(
                config=pipeline_config,
                alpha_scores=composite,
                stock_returns=returns,
                factor_returns=ff_no_rf,
                beta=beta,
                sector_labels=sectors,
                universe=universe,
                rf=rf,
                use_optimizer=False,
            )
            result = bt.run(start_date=pipeline_config.dates.burn_in_end,
                            end_date=pipeline_config.dates.end)
            net = result.net_returns

            if len(net) < 24:
                print(f"    No results")
                continue

            pre_oos = net[net.index < oos_start]
            oos_net = net[net.index >= oos_start]

            print(f"\n    Performance (annualized):")
            print(f"      Pre-OOS Sharpe:   {sharpe_ratio(pre_oos):>8.4f}")
            print(f"      OOS Sharpe:       {sharpe_ratio(oos_net):>8.4f}")
            print(f"      Full Sharpe:      {sharpe_ratio(net):>8.4f}")
            print(f"      Sortino:          {sortino_ratio(net):>8.4f}")
            print(f"      Max Drawdown:     {max_drawdown(net):>8.4f}")
            print(f"      Avg Turnover:     {result.turnover.mean():>8.4f}")

            # Factor attribution
            attr = factor_attribution(net, ff_no_rf, rf)
            print(f"\n    Factor Attribution:")
            print(f"      FF Alpha (ann):   {attr.get('alpha', 0):>8.4f}")
            print(f"      Alpha t-stat:     {attr.get('alpha_t_stat', 0):>8.2f}")
            print(f"      R-squared:        {attr.get('r_squared', 0):>8.4f}")

            # Bootstrap CI on OOS
            if len(oos_net) > 12:
                boot = block_bootstrap_sharpe(oos_net, n_bootstrap=10000)
                print(f"\n    Bootstrap 95% CI on OOS Sharpe:")
                print(f"      Sharpe: {boot['sharpe']:.4f}  CI [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

            # Long vs short leg
            long_rets = {}
            short_rets = {}
            for t, weights in result.holdings.items():
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
            if len(long_s) > 12 and len(short_s) > 12:
                print(f"\n    Leg Attribution:")
                print(f"      Long Sharpe:      {sharpe_ratio(long_s):>8.4f}")
                print(f"      Short Sharpe:     {sharpe_ratio(short_s):>8.4f}")

        logger.step_complete("Final Evaluation", f"{len(optimal_signals)} signals, EW + IC-weighted")

    elapsed = time.time() - start_time
    logger.pipeline_complete(f"Total {elapsed/60:.1f} min")
    print(f"\n{'='*80}")
    print(f"  FULL PIPELINE COMPLETE in {elapsed:.0f} seconds ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
