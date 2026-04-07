"""Unified pipeline: mine → pre-select → stepwise → optimal signal set.

Supports two modes:
- "single_split": Legacy fixed train/val/OOS split
- "nested": Desk-grade nested chronological validation with inner folds,
  cross-fold stability, model comparison, and OOS-only final evaluation
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.gpu.backend import GPU_AVAILABLE
from src.mining.config import MiningConfig
from src.mining.runner import run_mining, run_mining_nested
from src.mining.stepwise import forward_stepwise_selection
from src.mining.persistence import RunContext
from src.utils.logger import PipelineLogger


def main():
    pipeline_config = PipelineConfig.from_yaml("config/pipeline.yaml")
    mining_config = MiningConfig.from_yaml("config/mining.yaml")
    panel = DataPanel(pipeline_config)

    logger = PipelineLogger("full_pipeline")

    mode = pipeline_config.validation.mode
    print("=" * 80)
    print(f"  FULL PIPELINE — MODE: {mode.upper()}")
    print(f"  GPU: {'RTX 4080 SUPER (CuPy)' if GPU_AVAILABLE else 'Not available (CPU mode)'}")
    print("=" * 80)

    start_time = time.time()

    # Load shared data
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

    if mode == "nested":
        _run_nested_pipeline(
            pipeline_config, mining_config, panel, logger,
            returns, universe, log_mcap, beta, sectors,
            ff_no_rf, rf, start_time,
        )
    else:
        _run_single_split_pipeline(
            pipeline_config, mining_config, panel, logger,
            returns, universe, log_mcap, beta, sectors,
            ff_no_rf, rf, start_time,
        )


# ===========================================================================
# NESTED VALIDATION PIPELINE
# ===========================================================================

def _run_nested_pipeline(
    pipeline_config, mining_config, panel, logger,
    returns, universe, log_mcap, beta, sectors,
    ff_no_rf, rf, start_time,
):
    """Desk-grade nested chronological validation pipeline."""
    vc = pipeline_config.validation

    run_ctx = RunContext(pipeline_config.output_dir, pipeline_config.project_root)
    run_ctx.save_config_snapshot(pipeline_config, mining_config)

    logger.step_start("Nested Mining")

    # Run the full nested pipeline: enumerate → compute → inner folds → stability → stepwise → model comparison
    optimal_signals, selected_method, precomputed_neutral = run_mining_nested(
        pipeline_config, mining_config, run_ctx,
        returns, universe, sectors, log_mcap, beta,
        ff_no_rf, rf,
        logger=logger,
    )
    logger.step_complete("Nested Mining",
                         f"{len(optimal_signals)} signals, method={selected_method}")

    # Stage 7: OOS-only evaluation
    if optimal_signals:
        logger.step_start("Stage 7: OOS Evaluation")
        _run_stage7_oos(
            optimal_signals, selected_method, precomputed_neutral,
            pipeline_config, returns, universe, sectors, log_mcap, beta,
            ff_no_rf, rf, panel, run_ctx,
        )
        logger.step_complete("Stage 7: OOS Evaluation",
                             f"{len(optimal_signals)} signals on OOS")

    elapsed = time.time() - start_time
    logger.pipeline_complete(f"Total {elapsed/60:.1f} min")
    print(f"\n{'='*80}")
    print(f"  NESTED PIPELINE COMPLETE in {elapsed:.0f} seconds ({elapsed/60:.1f} min)")
    print(f"  Run artifacts: {run_ctx.root}")
    print(f"{'='*80}")


def _run_stage7_oos(
    optimal_signals, selected_method, precomputed_neutral,
    config, returns, universe, sectors, log_mcap, beta,
    ff_no_rf, rf, panel, run_ctx,
):
    """Stage 7: Final evaluation on OOS period ONLY (2015-2019).

    Backtest warm-up starts ~2010 for covariance estimation,
    but all reported metrics use only the OOS window.
    """
    from src.signals.combine import combine_signals
    from src.portfolio.backtest import WalkForwardBacktest
    from src.analytics.performance import (
        sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    )
    from src.analytics.risk import (
        parametric_var, historical_var, cvar, cornish_fisher_var, drawdown_stats,
    )
    from src.analytics.attribution import factor_attribution, print_attribution
    from src.analytics.bootstrap import block_bootstrap_sharpe
    from src.analytics.bias_aware import bias_aware_report
    from src.signals.zscore import standardize_signal as std_sig
    from src.signals.neutralize import neutralize_signal as neut_sig

    vc = config.validation
    oos_start = pd.Period(vc.oos_start, "M")
    oos_mid = pd.Period("2018-01", "M")

    # Warm-up: start backtest well before OOS for covariance estimation
    warmup_start = str(pd.Period(vc.middle_start, "M") - config.backtest.lookback_cov)

    print(f"\n{'='*80}")
    print(f"  STAGE 7: OOS-ONLY EVALUATION — {len(optimal_signals)} signals, method={selected_method}")
    print(f"  OOS period: {vc.oos_start} to {config.dates.end}")
    print(f"{'='*80}")

    # Get neutralized signals
    optimal_neutral = {n: precomputed_neutral[n] for n in optimal_signals
                       if n in precomputed_neutral}

    # --- 7a. Combine using selected method ---
    print(f"\n  7a. Combining signals ({selected_method})...")
    composite = combine_signals(
        optimal_neutral, method=selected_method,
        returns=returns, universe=universe,
        lookback=config.backtest.lookback_signal,
    )

    # --- 7b. Backtest (warm-up from ~2005, report only OOS) ---
    print(f"\n  7b. Walk-forward backtest (warm-up from {warmup_start})...")
    bt = WalkForwardBacktest(
        config=config, alpha_scores=composite, stock_returns=returns,
        factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
        universe=universe, rf=rf, use_optimizer=False,
    )
    result = bt.run(start_date=warmup_start, end_date=config.dates.end)
    net = result.net_returns

    # Filter to OOS only for metrics
    oos_net = net[net.index >= oos_start]

    if len(oos_net) < 12:
        print("  WARNING: Insufficient OOS data for evaluation")
        return

    # --- 7c. OOS Performance ---
    print(f"\n  {'='*70}")
    print(f"  7c. OOS Performance ({vc.oos_start} to {config.dates.end})")
    print(f"  {'='*70}")

    avg_to = result.turnover[result.turnover.index >= oos_start].mean()
    print(f"\n    OOS Sharpe:       {sharpe_ratio(oos_net):>8.4f}")
    print(f"    Ann. Return:      {oos_net.mean()*12:>8.4f}")
    print(f"    Ann. Vol:         {oos_net.std()*np.sqrt(12):>8.4f}")
    print(f"    Sortino:          {sortino_ratio(oos_net):>8.4f}")
    print(f"    Calmar:           {calmar_ratio(oos_net):>8.4f}")
    print(f"    Max Drawdown:     {max_drawdown(oos_net):>8.4f}")
    print(f"    Win Rate:         {(oos_net>0).mean():>8.4f}")
    print(f"    Avg Turnover:     {avg_to:>8.4f}")
    print(f"    N Months:         {len(oos_net):>8d}")

    # Risk
    dd = drawdown_stats(oos_net)
    print(f"\n    Risk:")
    print(f"      Parametric VaR:   {parametric_var(oos_net):>8.4f}")
    print(f"      Historical VaR:   {historical_var(oos_net):>8.4f}")
    print(f"      CVaR (ES):        {cvar(oos_net):>8.4f}")
    print(f"      CF-VaR:           {cornish_fisher_var(oos_net):>8.4f}")

    # --- 7d. Factor attribution on OOS ---
    print(f"\n  7d. Factor Attribution (OOS only):")
    oos_ff = ff_no_rf[ff_no_rf.index >= oos_start]
    oos_rf = rf[rf.index >= oos_start]
    attr = factor_attribution(oos_net, oos_ff, oos_rf)
    print_attribution(attr)

    # --- 7e. Bootstrap CI + p-value ---
    print(f"\n  7e. Bootstrap 95% CI on OOS Sharpe:")
    boot = block_bootstrap_sharpe(oos_net, n_bootstrap=10000)
    print(f"    Sharpe: {boot['sharpe']:.4f}  CI [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print(f"    p-value (SR>0): {boot['p_value_zero']:.4f}")

    # --- 7f. Bias-aware statistics (DSR, PBO) ---
    print(f"\n  7f. Selection-Bias-Aware Statistics:")
    n_trials = 770  # approximate number of candidates tested
    bias = bias_aware_report(oos_net, n_trials=n_trials)
    print(f"    Deflated Sharpe Ratio (DSR): {bias['dsr']:.4f}")
    print(f"    N trials corrected for:      {n_trials}")
    if not np.isnan(bias.get('pbo', np.nan)):
        print(f"    Prob. Backtest Overfitting:   {bias['pbo']:.4f}")

    # --- 7g. Subperiod stability (OOS only) ---
    print(f"\n  {'='*70}")
    print(f"  7g. OOS Subperiod Stability")
    print(f"  {'='*70}")
    periods = {
        "OOS Early (2015-2017)": (oos_start, pd.Period("2017-12", "M")),
        "OOS Late (2018-2019)": (oos_mid, pd.Period("2019-12", "M")),
        "Full OOS (2015-2019)": (oos_start, pd.Period("2019-12", "M")),
    }
    print(f"\n    {'Period':30s} {'Sharpe':>8s} {'Return':>8s} {'Vol':>8s} {'MaxDD':>8s} {'Months':>8s}")
    print(f"    {'-'*72}")
    for pname, (pstart, pend) in periods.items():
        sub = oos_net[(oos_net.index >= pstart) & (oos_net.index <= pend)]
        if len(sub) < 6:
            continue
        print(f"    {pname:30s} {sharpe_ratio(sub):>8.4f} {sub.mean()*12:>8.4f} "
              f"{sub.std()*np.sqrt(12):>8.4f} {max_drawdown(sub):>8.4f} {len(sub):>8d}")

    # --- 7h. Cost sensitivity (OOS only) ---
    print(f"\n  {'='*70}")
    print(f"  7h. Cost Sensitivity (OOS)")
    print(f"  {'='*70}")
    print(f"\n    {'Cost (bps RT)':>15s} {'OOS SR':>8s}")
    print(f"    {'-'*25}")
    for cost_bps in [0, 5, 10, 15, 20, 30]:
        cfg_copy = deepcopy(config)
        cfg_copy.costs.base_bps = cost_bps
        bt_cost = WalkForwardBacktest(
            config=cfg_copy, alpha_scores=composite, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        res_cost = bt_cost.run(start_date=warmup_start, end_date=config.dates.end)
        oos_c = res_cost.net_returns[res_cost.net_returns.index >= oos_start]
        print(f"    {cost_bps:>15d} {sharpe_ratio(oos_c):>8.4f}")

    # --- 7i. Regime analysis (OOS only) ---
    print(f"\n  {'='*70}")
    print(f"  7i. Regime Analysis (OOS, Bull vs Bear)")
    print(f"  {'='*70}")
    sp500_ret = panel.get_market_excess()
    sp500_trailing = sp500_ret.rolling(12).sum()
    bull = sp500_trailing[sp500_trailing > 0].index
    bear = sp500_trailing[sp500_trailing <= 0].index
    net_bull = oos_net[oos_net.index.isin(bull)]
    net_bear = oos_net[oos_net.index.isin(bear)]
    print(f"\n    {'Regime':15s} {'Sharpe':>8s} {'Return':>8s} {'Months':>8s}")
    print(f"    {'-'*40}")
    if len(net_bull) > 6:
        print(f"    {'Bull':15s} {sharpe_ratio(net_bull):>8.4f} {net_bull.mean()*12:>8.4f} {len(net_bull):>8d}")
    if len(net_bear) > 6:
        print(f"    {'Bear':15s} {sharpe_ratio(net_bear):>8.4f} {net_bear.mean()*12:>8.4f} {len(net_bear):>8d}")

    # --- 7j. Signal ablation (OOS only) ---
    print(f"\n  {'='*70}")
    print(f"  7j. Signal Ablation (OOS)")
    print(f"  {'='*70}")
    baseline_oos_sr = sharpe_ratio(oos_net)
    print(f"\n    {'Dropped':30s} {'OOS SR':>10s} {'Delta':>8s}")
    print(f"    {'-'*50}")
    print(f"    {'(none — baseline)':30s} {baseline_oos_sr:>10.4f}")
    for drop_name in optimal_signals:
        remaining_sigs = {n: s for n, s in optimal_neutral.items() if n != drop_name}
        if not remaining_sigs:
            continue
        abl_composite = combine_signals(remaining_sigs, method="equal")
        bt_abl = WalkForwardBacktest(
            config=config, alpha_scores=abl_composite, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        res_abl = bt_abl.run(start_date=warmup_start, end_date=config.dates.end)
        abl_oos = res_abl.net_returns[res_abl.net_returns.index >= oos_start]
        abl_sr = sharpe_ratio(abl_oos) if len(abl_oos) > 6 else 0
        delta = abl_sr - baseline_oos_sr
        print(f"    {drop_name:30s} {abl_sr:>10.4f} {delta:>+8.4f}")

    # --- 7k. Quantile monotonicity (OOS only) ---
    print(f"\n  {'='*70}")
    print(f"  7k. Quantile Monotonicity (OOS decile returns)")
    print(f"  {'='*70}")
    decile_rets = {q: [] for q in range(1, 11)}
    oos_dates = composite.dropna(how="all").index
    oos_dates = oos_dates[oos_dates >= oos_start]
    for t in oos_dates:
        t1 = t + 1
        if t1 not in returns.index or t not in universe.index:
            continue
        univ_t = universe.loc[t]
        members = univ_t[univ_t].index
        alpha_t = composite.loc[t].reindex(members).dropna()
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
    else:
        print()

    # Save OOS evaluation artifacts
    oos_metrics = {
        "performance": {
            "oos_sharpe": sharpe_ratio(oos_net),
            "ann_return": float(oos_net.mean() * 12),
            "ann_vol": float(oos_net.std() * np.sqrt(12)),
            "sortino": sortino_ratio(oos_net),
            "max_drawdown": max_drawdown(oos_net),
            "n_months": len(oos_net),
            "selected_method": selected_method,
        },
        "bias_aware": bias,
        "bootstrap": boot,
        "attribution": attr,
    }
    run_ctx.save_oos_evaluation(oos_metrics)
    print(f"\n  OOS evaluation artifacts saved to {run_ctx.root / 'oos_evaluation'}")


# ===========================================================================
# LEGACY SINGLE-SPLIT PIPELINE
# ===========================================================================

def _run_single_split_pipeline(
    pipeline_config, mining_config, panel, logger,
    returns, universe, log_mcap, beta, sectors,
    ff_no_rf, rf, start_time,
):
    """Legacy single-split validation pipeline (original behavior)."""
    from src.signals.zscore import standardize_signal as std_sig
    from src.signals.neutralize import neutralize_signal as neut_sig, verify_neutralization
    from src.signals.combine import combine_signals as combine_final
    from src.signals.report_card import signal_report_card, print_report_card
    from src.portfolio.backtest import WalkForwardBacktest
    from src.analytics.performance import (
        sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    )
    from src.analytics.risk import (
        parametric_var, historical_var, cvar, cornish_fisher_var, drawdown_stats,
    )
    from src.analytics.attribution import factor_attribution, print_attribution
    from src.analytics.bootstrap import block_bootstrap_sharpe
    from src.analytics.ic import compute_ic_series, ic_summary, ic_decay_analysis

    logger.step_start("Mining")

    try:
        run_mining(pipeline_config, mining_config)
        logger.step_complete("Mining", "Enumerated, evaluated, filtered, deduped")
    except Exception as e:
        logger.step_failed("Mining", str(e))
        raise

    # Load candidates
    print(f"\n{'='*80}")
    print("  FORWARD STEPWISE PORTFOLIO SELECTION")
    print(f"{'='*80}")

    candidates = {}
    mined_dir = Path(pipeline_config.project_root) / mining_config.factor_dir
    for py_file in sorted(mined_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        signal_name = py_file.stem
        cache_file = pipeline_config.cache_path(f"raw_{signal_name}.parquet")
        if cache_file.exists():
            candidates[signal_name] = pd.read_parquet(cache_file)

    # Also load any active signals from config
    for name in pipeline_config.signals.active:
        cache_file = pipeline_config.cache_path(f"raw_{name}.parquet")
        if cache_file.exists():
            candidates[name] = pd.read_parquet(cache_file)

    logger.info(f"Total candidates for stepwise: {len(candidates)}")
    logger.step_start("Stepwise Selection")

    # Build projection cache
    from src.gpu.neutralize_batch import ProjectionCache
    proj_cache = ProjectionCache()
    proj_cache.build(sector_labels=sectors, log_mcap=log_mcap, beta=beta)

    history = forward_stepwise_selection(
        candidate_signals=candidates,
        returns=returns, universe=universe,
        sector_labels=sectors, log_mcap=log_mcap, beta=beta,
        factor_returns=ff_no_rf, rf=rf,
        config=pipeline_config,
        eval_metric="full_sharpe",
        max_signals=15, min_improvement=0.01,
        projection_cache=proj_cache,
    )

    if history:
        rows = [{
            "step": r.step, "added_signal": r.added_signal,
            "n_signals": r.n_signals, "full_sharpe": r.full_sharpe,
            "oos_sharpe": r.oos_sharpe, "ff_alpha": r.ff_alpha,
            "ff_alpha_t": r.ff_alpha_t, "improvement": r.improvement,
            "signal_set": "|".join(r.signal_set),
        } for r in history]
        pd.DataFrame(rows).to_csv(
            pipeline_config.cache_path("stepwise_v2_results.csv"), index=False
        )

    if history:
        best = max(history, key=lambda r: r.full_sharpe)
        logger.step_complete("Stepwise Selection",
                             f"{best.n_signals} signals, SR={best.full_sharpe:.4f}")
        optimal_signals = best.signal_set
    else:
        logger.step_complete("Stepwise Selection", "No signals selected")
        optimal_signals = []

    # Stage 7: Full evaluation (legacy — runs on all periods)
    if optimal_signals:
        print(f"\n{'='*80}")
        print(f"  STAGE 7: FINAL EVALUATION — {len(optimal_signals)} SIGNALS")
        print(f"{'='*80}")
        logger.step_start("Stage 7: Final Evaluation")

        oos_start = pd.Period("2015-01", "M")

        # Neutralize
        optimal_raw = {name: candidates[name] for name in optimal_signals if name in candidates}
        optimal_neutral = {}
        for name, raw in optimal_raw.items():
            z = std_sig(raw, winsorize_pct=0.01)
            n = neut_sig(z, sector_labels=sectors, log_mcap=log_mcap, beta=beta,
                         neutralize_sector=True, neutralize_size=True, neutralize_beta=True)
            optimal_neutral[name] = n

        # Combination + backtest
        composite_ew = combine_final(optimal_neutral, method="equal")

        bt = WalkForwardBacktest(
            config=pipeline_config, alpha_scores=composite_ew, stock_returns=returns,
            factor_returns=ff_no_rf, beta=beta, sector_labels=sectors,
            universe=universe, rf=rf, use_optimizer=False,
        )
        result = bt.run(start_date=pipeline_config.dates.burn_in_end,
                        end_date=pipeline_config.dates.end)
        net = result.net_returns

        if len(net) > 24:
            oos_net = net[net.index >= oos_start]
            pre_oos = net[net.index < oos_start]

            print(f"\n    Pre-OOS Sharpe: {sharpe_ratio(pre_oos):.4f}")
            print(f"    OOS Sharpe:     {sharpe_ratio(oos_net):.4f}")
            print(f"    Full Sharpe:    {sharpe_ratio(net):.4f}")

            attr = factor_attribution(net, ff_no_rf, rf)
            print_attribution(attr)

            if len(oos_net) > 12:
                boot = block_bootstrap_sharpe(oos_net, n_bootstrap=10000)
                print(f"    Bootstrap OOS Sharpe: {boot['sharpe']:.4f} "
                      f"CI [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        logger.step_complete("Stage 7: Final Evaluation",
                             f"{len(optimal_signals)} signals")

    elapsed = time.time() - start_time
    logger.pipeline_complete(f"Total {elapsed/60:.1f} min")
    print(f"\n{'='*80}")
    print(f"  PIPELINE COMPLETE in {elapsed:.0f} seconds ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
