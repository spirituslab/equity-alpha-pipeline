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
    # STAGE 7: FINAL EVALUATION — Full pipeline on optimal signals
    # Uses EVERYTHING we built: IC-weighted, constrained optimizer,
    # factor attribution, risk metrics, robustness, signal report cards
    # ================================================================
    if optimal_signals:
        print(f"\n{'='*80}")
        print(f"  STAGE 7: FINAL EVALUATION — FULL PIPELINE ON OPTIMAL {len(optimal_signals)} SIGNALS")
        print(f"{'='*80}")
        logger.step_start("Stage 7: Final Evaluation")

        import numpy as np
        from copy import deepcopy
        from src.signals.zscore import standardize_signal as std_sig
        from src.signals.neutralize import neutralize_signal as neut_sig, verify_neutralization
        from src.signals.combine import combine_signals as combine_final
        from src.signals.report_card import signal_report_card, print_report_card
        from src.portfolio.backtest import WalkForwardBacktest
        from src.portfolio.optimization import long_short_optimize
        from src.portfolio.risk_model import FactorRiskModel
        from src.analytics.performance import (
            sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
            cumulative_returns, rolling_sharpe, compute_descriptive_stats,
        )
        from src.analytics.risk import (
            parametric_var, historical_var, cvar, cornish_fisher_var, drawdown_stats,
        )
        from src.analytics.attribution import factor_attribution, print_attribution
        from src.analytics.bootstrap import block_bootstrap_sharpe
        from src.analytics.statistical_tests import sharpe_ratio_test
        from src.analytics.ic import compute_ic_series, ic_summary, ic_decay_analysis

        oos_start = pd.Period("2015-01", "M")
        oos_mid = pd.Period("2018-01", "M")

        # --- 7a. Neutralize (full OLS) ---
        print(f"\n  7a. Neutralizing {len(optimal_signals)} signals (full OLS)...")
        optimal_raw = {name: candidates[name] for name in optimal_signals if name in candidates}
        optimal_neutral = {}
        optimal_z = {}
        for name, raw in optimal_raw.items():
            z = std_sig(raw, winsorize_pct=0.01)
            optimal_z[name] = z
            n = neut_sig(z, sector_labels=sectors, log_mcap=log_mcap, beta=beta,
                         neutralize_sector=True, neutralize_size=True, neutralize_beta=True)
            optimal_neutral[name] = n

        # Verify neutralization
        print(f"\n  7a. Neutralization verification:")
        for name, sig in optimal_neutral.items():
            quality = verify_neutralization(sig, sectors, log_mcap, beta)
            print(f"    {name}: size={quality.get('avg_abs_corr_size',0):.4f}, "
                  f"beta={quality.get('avg_abs_corr_beta',0):.4f}, "
                  f"sector={quality.get('avg_abs_corr_sector',0):.4f}")

        # --- 7b. Signal report cards ---
        print(f"\n  7b. Signal report cards:")
        for name, sig in optimal_neutral.items():
            others = {n: s for n, s in optimal_neutral.items() if n != name}
            report = signal_report_card(name, sig, returns, universe,
                                         other_signals=others, end_date=pipeline_config.dates.val_end)
            print_report_card(report)

        # --- 7c. IC decay analysis ---
        print(f"\n  7c. IC decay (signal persistence):")
        print(f"    {'Signal':30s} {'Lag1':>7s} {'Lag3':>7s} {'Lag6':>7s} {'Lag12':>7s}")
        for name, sig in optimal_neutral.items():
            decay = ic_decay_analysis(sig, returns, universe, max_lag=12, end_date=pipeline_config.dates.end)
            if not decay.empty:
                l1 = decay.loc[1, "Mean IC"] if 1 in decay.index else 0
                l3 = decay.loc[3, "Mean IC"] if 3 in decay.index else 0
                l6 = decay.loc[6, "Mean IC"] if 6 in decay.index else 0
                l12 = decay.loc[12, "Mean IC"] if 12 in decay.index else 0
                print(f"    {name:30s} {l1:>7.4f} {l3:>7.4f} {l6:>7.4f} {l12:>7.4f}")

        # --- 7d. Combination + backtest ---
        composite_ew = combine_final(optimal_neutral, method="equal")
        composite_ic = combine_final(optimal_neutral, method="ic_weighted",
                                      returns=returns, universe=universe,
                                      lookback=pipeline_config.backtest.lookback_signal)
        composite_iv = combine_final(optimal_neutral, method="inverse_vol",
                                      returns=returns, universe=universe,
                                      lookback=pipeline_config.backtest.lookback_signal)

        all_composites = [
            ("Equal-Weight", composite_ew),
            ("IC-Weighted", composite_ic),
            ("Inverse-Vol", composite_iv),
        ]

        for combo_name, composite in all_composites:
            print(f"\n  {'='*70}")
            print(f"  7d. {combo_name} — Naive L/S Backtest")
            print(f"  {'='*70}")

            bt = WalkForwardBacktest(
                config=pipeline_config, alpha_scores=composite, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=False,
            )
            result = bt.run(start_date=pipeline_config.dates.burn_in_end,
                            end_date=pipeline_config.dates.end)
            net = result.net_returns
            gross = result.gross_returns

            if len(net) < 24:
                continue

            pre_oos = net[net.index < oos_start]
            oos_net = net[net.index >= oos_start]

            # Performance
            avg_to = result.turnover.mean()
            avg_holding = 2.0 / avg_to if avg_to > 0 else np.nan  # approx holding period in months
            print(f"\n    Performance:")
            print(f"      Pre-OOS Sharpe:   {sharpe_ratio(pre_oos):>8.4f}")
            print(f"      OOS Sharpe:       {sharpe_ratio(oos_net):>8.4f}")
            print(f"      Full Sharpe:      {sharpe_ratio(net):>8.4f}")
            print(f"      Ann. Return:      {net.mean()*12:>8.4f}")
            print(f"      Ann. Vol:         {net.std()*np.sqrt(12):>8.4f}")
            print(f"      Sortino:          {sortino_ratio(net):>8.4f}")
            print(f"      Calmar:           {calmar_ratio(net):>8.4f}")
            print(f"      Max Drawdown:     {max_drawdown(net):>8.4f}")
            print(f"      Win Rate:         {(net>0).mean():>8.4f}")
            print(f"      Skewness:         {pd.Series(net).skew():>8.4f}")
            print(f"      Kurtosis:         {pd.Series(net).kurtosis():>8.4f}")
            print(f"      Avg Turnover:     {avg_to:>8.4f}")
            print(f"      Avg Holding Per.: {avg_holding:>8.1f} months")
            print(f"      Avg Gross Exp.:   {np.mean([w.abs().sum() for w in result.holdings.values()]):>8.4f}")
            print(f"      Avg Net Exp.:     {np.mean([w.sum() for w in result.holdings.values()]):>8.4f}")

            # Risk metrics
            dd = drawdown_stats(net)
            print(f"\n    Risk:")
            print(f"      Parametric VaR:   {parametric_var(net):>8.4f}")
            print(f"      Historical VaR:   {historical_var(net):>8.4f}")
            print(f"      CVaR (ES):        {cvar(net):>8.4f}")
            print(f"      CF-VaR:           {cornish_fisher_var(net):>8.4f}")
            print(f"      Avg Drawdown:     {dd.get('Avg Drawdown',0):>8.4f}")
            print(f"      Max DD Duration:  {dd.get('Max Drawdown Duration (months)',0):>8.0f} months")

            # Factor attribution (full)
            attr = factor_attribution(net, ff_no_rf, rf)
            print_attribution(attr)

            # Bootstrap CI
            if len(oos_net) > 12:
                boot = block_bootstrap_sharpe(oos_net, n_bootstrap=10000)
                print(f"    Bootstrap 95% CI on OOS Sharpe:")
                print(f"      Sharpe: {boot['sharpe']:.4f}  CI [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

            boot_full = block_bootstrap_sharpe(net, n_bootstrap=10000)
            print(f"    Bootstrap 95% CI on Full Sharpe:")
            print(f"      Sharpe: {boot_full['sharpe']:.4f}  CI [{boot_full['ci_lower']:.4f}, {boot_full['ci_upper']:.4f}]")

            # Long vs short leg
            long_rets, short_rets = {}, {}
            for t, weights in result.holdings.items():
                t1 = t + 1
                if t1 not in returns.index:
                    continue
                r = returns.loc[t1]
                lw = weights[weights > 0.001]
                sw = weights[weights < -0.001]
                if len(lw) > 0:
                    long_rets[t1] = (lw * r.reindex(lw.index).fillna(0)).sum() / lw.sum()
                if len(sw) > 0:
                    short_rets[t1] = (sw * r.reindex(sw.index).fillna(0)).sum() / abs(sw.sum())

            long_s, short_s = pd.Series(long_rets), pd.Series(short_rets)
            if len(long_s) > 12:
                print(f"\n    Long/Short Leg:")
                print(f"      Long Sharpe:      {sharpe_ratio(long_s):>8.4f}")
                print(f"      Short Sharpe:     {sharpe_ratio(short_s):>8.4f}")
                print(f"      Long Ann. Return: {long_s.mean()*12:>8.4f}")
                print(f"      Short Ann. Return:{short_s.mean()*12:>8.4f}")

        # --- Statistical comparison: EW vs IC-weighted ---
        from src.analytics.statistical_tests import diebold_mariano_test
        bt_ew_ref = WalkForwardBacktest(
            config=pipeline_config, alpha_scores=composite_ew, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        res_ew_ref = bt_ew_ref.run(start_date=pipeline_config.dates.burn_in_end,
                                    end_date=pipeline_config.dates.end)
        bt_ic_ref = WalkForwardBacktest(
            config=pipeline_config, alpha_scores=composite_ic, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        res_ic_ref = bt_ic_ref.run(start_date=pipeline_config.dates.burn_in_end,
                                    end_date=pipeline_config.dates.end)

        print(f"\n  Statistical Tests: EW vs IC-Weighted")
        sr_test = sharpe_ratio_test(res_ew_ref.net_returns, res_ic_ref.net_returns)
        print(f"    Sharpe Equality Test: z={sr_test['z-statistic']:.2f}, p={sr_test['p-value']:.4f}")
        dm_test = diebold_mariano_test(res_ew_ref.net_returns, res_ic_ref.net_returns)
        print(f"    Diebold-Mariano Test: DM={dm_test['DM Statistic']:.2f}, p={dm_test['p-value']:.4f}")

        # Mean return significance
        from src.analytics.performance import t_test_mean
        t_ew = t_test_mean(res_ew_ref.net_returns)
        print(f"    EW Mean Return t-test: t={t_ew['t_stat']:.2f}, p={t_ew['p_value']:.4f}")

        # --- 7e. Constrained optimizer backtest (all combination methods) ---
        print(f"\n  {'='*70}")
        print(f"  7e. Constrained Optimizer Backtest")
        print(f"  {'='*70}")

        for combo_name, composite in all_composites:
            print(f"\n    --- Optimizer + {combo_name} ---")
            bt_opt = WalkForwardBacktest(
                config=pipeline_config, alpha_scores=composite, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=True,
            )
            result_opt = bt_opt.run(start_date=pipeline_config.dates.burn_in_end,
                                    end_date=pipeline_config.dates.end)
            net_opt = result_opt.net_returns

            if len(net_opt) > 24:
                pre_oos_opt = net_opt[net_opt.index < oos_start]
                oos_opt = net_opt[net_opt.index >= oos_start]
                print(f"      Pre-OOS Sharpe:   {sharpe_ratio(pre_oos_opt):>8.4f}")
                print(f"      OOS Sharpe:       {sharpe_ratio(oos_opt):>8.4f}")
                print(f"      Sortino:          {sortino_ratio(net_opt):>8.4f}")
                print(f"      Max Drawdown:     {max_drawdown(net_opt):>8.4f}")
                print(f"      Avg |Beta|:       {result_opt.realized_beta.abs().mean():>8.4f}")
                print(f"      Avg Turnover:     {result_opt.turnover.mean():>8.4f}")
                attr_opt = factor_attribution(net_opt, ff_no_rf, rf)
                print(f"      FF Alpha (ann):   {attr_opt.get('alpha',0):>8.4f}")
                print(f"      Alpha t-stat:     {attr_opt.get('alpha_t_stat',0):>8.2f}")

        # --- 7f. Robustness: subperiod stability ---
        print(f"\n  {'='*70}")
        print(f"  7f. Subperiod Stability")
        print(f"  {'='*70}")

        # Use equal-weight naive L/S net returns
        net_final = result.net_returns  # from last backtest in the loop (IC-weighted or EW)
        # Re-run EW for clean reference
        bt_ew = WalkForwardBacktest(
            config=pipeline_config, alpha_scores=composite_ew, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        result_ew = bt_ew.run(start_date=pipeline_config.dates.burn_in_end,
                              end_date=pipeline_config.dates.end)
        net_ew = result_ew.net_returns

        periods = {
            "Development (1975-2004)": (pd.Period("1975-01", "M"), pd.Period("2004-12", "M")),
            "Validation (2005-2014)": (pd.Period("2005-01", "M"), pd.Period("2014-12", "M")),
            "OOS Early (2015-2017)": (oos_start, pd.Period("2017-12", "M")),
            "OOS Late (2018-2019)": (oos_mid, pd.Period("2019-12", "M")),
            "Full OOS (2015-2019)": (oos_start, pd.Period("2019-12", "M")),
        }
        print(f"\n    {'Period':30s} {'Sharpe':>8s} {'Return':>8s} {'Vol':>8s} {'MaxDD':>8s} {'Months':>8s}")
        print(f"    {'-'*72}")
        for pname, (pstart, pend) in periods.items():
            sub = net_ew[(net_ew.index >= pstart) & (net_ew.index <= pend)]
            if len(sub) < 6:
                continue
            print(f"    {pname:30s} {sharpe_ratio(sub):>8.4f} {sub.mean()*12:>8.4f} "
                  f"{sub.std()*np.sqrt(12):>8.4f} {max_drawdown(sub):>8.4f} {len(sub):>8d}")

        # --- 7g. Cost sensitivity ---
        print(f"\n  {'='*70}")
        print(f"  7g. Cost Sensitivity")
        print(f"  {'='*70}")
        print(f"\n    {'Cost (bps RT)':>15s} {'Full SR':>8s} {'OOS SR':>8s}")
        print(f"    {'-'*35}")
        for cost_bps in [0, 5, 10, 15, 20, 30]:
            cfg_copy = deepcopy(pipeline_config)
            cfg_copy.costs.base_bps = cost_bps
            bt_cost = WalkForwardBacktest(
                config=cfg_copy, alpha_scores=composite_ew, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=False,
            )
            res_cost = bt_cost.run(start_date=pipeline_config.dates.burn_in_end,
                                    end_date=pipeline_config.dates.end)
            nc = res_cost.net_returns
            oos_c = nc[nc.index >= oos_start]
            print(f"    {cost_bps:>15d} {sharpe_ratio(nc):>8.4f} {sharpe_ratio(oos_c):>8.4f}")

        # --- 7h. Regime analysis ---
        print(f"\n  {'='*70}")
        print(f"  7h. Regime Analysis (Bull vs Bear)")
        print(f"  {'='*70}")
        sp500_ret = panel.get_market_excess()
        sp500_trailing = sp500_ret.rolling(12).sum()
        bull = sp500_trailing[sp500_trailing > 0].index
        bear = sp500_trailing[sp500_trailing <= 0].index
        net_bull = net_ew[net_ew.index.isin(bull)]
        net_bear = net_ew[net_ew.index.isin(bear)]
        print(f"\n    {'Regime':15s} {'Sharpe':>8s} {'Return':>8s} {'Months':>8s}")
        print(f"    {'-'*40}")
        if len(net_bull) > 6:
            print(f"    {'Bull':15s} {sharpe_ratio(net_bull):>8.4f} {net_bull.mean()*12:>8.4f} {len(net_bull):>8d}")
        if len(net_bear) > 6:
            print(f"    {'Bear':15s} {sharpe_ratio(net_bear):>8.4f} {net_bear.mean()*12:>8.4f} {len(net_bear):>8d}")

        # --- 7i. Signal ablation ---
        print(f"\n  {'='*70}")
        print(f"  7i. Signal Ablation")
        print(f"  {'='*70}")
        baseline_sr = sharpe_ratio(pre_oos)
        print(f"\n    {'Dropped':30s} {'Pre-OOS SR':>12s} {'Delta':>8s}")
        print(f"    {'-'*52}")
        print(f"    {'(none — baseline)':30s} {baseline_sr:>12.4f}")
        for drop_name in optimal_signals:
            remaining_sigs = {n: s for n, s in optimal_neutral.items() if n != drop_name}
            if not remaining_sigs:
                continue
            abl_composite = combine_final(remaining_sigs, method="equal")
            bt_abl = WalkForwardBacktest(
                config=pipeline_config, alpha_scores=abl_composite, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=False,
            )
            res_abl = bt_abl.run(start_date=pipeline_config.dates.burn_in_end,
                                  end_date=pipeline_config.dates.end)
            abl_pre = res_abl.net_returns[res_abl.net_returns.index < oos_start]
            abl_sr = sharpe_ratio(abl_pre)
            delta = abl_sr - baseline_sr
            print(f"    {drop_name:30s} {abl_sr:>12.4f} {delta:>+8.4f}")

        # --- 7j. Sleeve attribution (per-signal P&L contribution) ---
        print(f"\n  {'='*70}")
        print(f"  7j. Sleeve Attribution (per-signal contribution)")
        print(f"  {'='*70}")
        print(f"\n    {'Signal':30s} {'Sharpe':>8s} {'Ann. Return':>12s}")
        print(f"    {'-'*52}")
        for name, sig in optimal_neutral.items():
            # Backtest this signal alone
            bt_sleeve = WalkForwardBacktest(
                config=pipeline_config, alpha_scores=sig, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=False,
            )
            res_sleeve = bt_sleeve.run(start_date=pipeline_config.dates.burn_in_end,
                                        end_date=pipeline_config.dates.end)
            sleeve_net = res_sleeve.net_returns
            if len(sleeve_net) > 12:
                print(f"    {name:30s} {sharpe_ratio(sleeve_net):>8.4f} {sleeve_net.mean()*12:>12.4f}")

        # --- 7k. Sector attribution ---
        print(f"\n  {'='*70}")
        print(f"  7k. Sector Attribution")
        print(f"  {'='*70}")
        sector_rets = {}
        for t, weights in result_ew.holdings.items():
            t1 = t + 1
            if t1 not in returns.index or t not in sectors.index:
                continue
            r = returns.loc[t1]
            sec = sectors.loc[t]
            for stock, w in weights.items():
                if abs(w) < 0.0001:
                    continue
                s = sec.get(stock, "Other") if hasattr(sec, 'get') else "Other"
                if s not in sector_rets:
                    sector_rets[s] = []
                ret_contrib = w * r.get(stock, 0) if hasattr(r, 'get') else 0
                sector_rets[s].append(ret_contrib)

        print(f"\n    {'Sector':25s} {'Avg Monthly Contrib':>20s} {'Ann. Contrib':>15s}")
        print(f"    {'-'*62}")
        for sec_name in sorted(sector_rets.keys()):
            monthly = np.mean(sector_rets[sec_name])
            print(f"    {sec_name:25s} {monthly:>20.6f} {monthly*12:>15.4f}")

        # --- 7l. Neutralization sensitivity ---
        print(f"\n  {'='*70}")
        print(f"  7l. Neutralization Sensitivity")
        print(f"  {'='*70}")
        neut_configs = [
            ("None", False, False, False),
            ("Sector only", True, False, False),
            ("Size only", False, True, False),
            ("Beta only", False, False, True),
            ("Full (S+S+B)", True, True, True),
        ]
        print(f"\n    {'Neutralization':20s} {'Pre-OOS SR':>12s} {'OOS SR':>8s} {'FF α t':>8s}")
        print(f"    {'-'*50}")
        for nname, n_sec, n_size, n_beta in neut_configs:
            neut_sigs = {}
            for sname, raw in optimal_raw.items():
                z = std_sig(raw, winsorize_pct=0.01)
                neut_sigs[sname] = neut_sig(
                    z, sector_labels=sectors if n_sec else pd.DataFrame(),
                    log_mcap=log_mcap if n_size else pd.DataFrame(),
                    beta=beta if n_beta else pd.DataFrame(),
                    neutralize_sector=n_sec, neutralize_size=n_size, neutralize_beta=n_beta,
                )
            alpha_neut = combine_final(neut_sigs, method="equal")
            bt_neut = WalkForwardBacktest(
                config=pipeline_config, alpha_scores=alpha_neut, stock_returns=returns,
                factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
                universe=universe, rf=rf, use_optimizer=False,
            )
            res_neut = bt_neut.run(start_date=pipeline_config.dates.burn_in_end,
                                    end_date=pipeline_config.dates.end)
            nn = res_neut.net_returns
            pre_nn = nn[nn.index < oos_start]
            oos_nn = nn[nn.index >= oos_start]
            attr_nn = factor_attribution(nn, ff_no_rf, rf)
            print(f"    {nname:20s} {sharpe_ratio(pre_nn):>12.4f} {sharpe_ratio(oos_nn):>8.4f} "
                  f"{attr_nn.get('alpha_t_stat',0):>8.2f}")

        # --- 7m. Quantile monotonicity ---
        print(f"\n  {'='*70}")
        print(f"  7m. Quantile Monotonicity (decile returns)")
        print(f"  {'='*70}")
        # Use EW composite
        decile_rets = {q: [] for q in range(1, 11)}
        for t in composite_ew.dropna(how="all").index:
            t1 = t + 1
            if t1 not in returns.index or t not in universe.index:
                continue
            univ_t = universe.loc[t]
            members = univ_t[univ_t].index
            alpha_t = composite_ew.loc[t].reindex(members).dropna()
            ret_t = returns.loc[t1].reindex(members).dropna()
            common = alpha_t.index.intersection(ret_t.index)
            if len(common) < 50:
                continue
            alpha_sorted = alpha_t.loc[common].sort_values()
            n_per_q = len(common) // 10
            for q in range(1, 11):
                start_idx = (q - 1) * n_per_q
                end_idx = q * n_per_q if q < 10 else len(common)
                q_stocks = alpha_sorted.index[start_idx:end_idx]
                decile_rets[q].append(ret_t.loc[q_stocks].mean())

        print(f"\n    {'Decile':>8s} {'Avg Monthly Ret':>16s} {'Ann. Return':>12s}")
        print(f"    {'-'*38}")
        for q in range(1, 11):
            if decile_rets[q]:
                avg = np.mean(decile_rets[q])
                print(f"    {q:>8d} {avg:>16.4f} {avg*12:>12.4f}")
        print(f"    {'L/S (10-1)':>8s}", end="")
        if decile_rets[10] and decile_rets[1]:
            spread = np.mean(decile_rets[10]) - np.mean(decile_rets[1])
            print(f" {spread:>16.4f} {spread*12:>12.4f}")

        logger.step_complete("Stage 7: Final Evaluation",
                             f"{len(optimal_signals)} signals, full pipeline")

    elapsed = time.time() - start_time
    logger.pipeline_complete(f"Total {elapsed/60:.1f} min")
    print(f"\n{'='*80}")
    print(f"  FULL PIPELINE COMPLETE in {elapsed:.0f} seconds ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
