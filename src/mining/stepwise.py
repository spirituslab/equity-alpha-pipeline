"""Forward stepwise signal selection at the portfolio level.

Precomputes ALL neutralized signals once, then stepwise only does
combine + backtest per candidate (~1 sec instead of ~40 sec).

Includes both single-split (legacy) and nested multi-fold selection.
Candidate evaluations within each step are parallelized across CPU cores.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import multiprocessing
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig
from src.signals.zscore import standardize_signal
from src.signals.neutralize import neutralize_signal
from src.signals.combine import combine_signals
from src.portfolio.backtest import WalkForwardBacktest
from src.analytics.performance import sharpe_ratio
from src.analytics.attribution import factor_attribution

# Use fork on Linux to share memory via copy-on-write
_MP_CONTEXT = multiprocessing.get_context("fork")
_MAX_WORKERS = min(os.cpu_count() or 4, 16)
from src.gpu.backend import GPU_AVAILABLE


@dataclass
class StepResult:
    step: int
    added_signal: str
    signal_set: list[str]
    full_sharpe: float
    oos_sharpe: float
    ff_alpha: float
    ff_alpha_t: float
    n_signals: int
    improvement: float


def forward_stepwise_selection(
    candidate_signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    sector_labels: pd.DataFrame,
    log_mcap: pd.DataFrame,
    beta: pd.DataFrame,
    factor_returns: pd.DataFrame,
    rf: pd.Series,
    config: PipelineConfig,
    eval_metric: str = "full_sharpe",
    max_signals: int = 15,
    min_improvement: float = 0.01,
    projection_cache=None,
) -> list[StepResult]:
    """Forward stepwise signal selection via portfolio-level backtesting.

    Key optimization: precompute ALL neutralized signals once upfront.
    Each stepwise trial then only does combine + backtest (~1 sec).
    """
    oos_start = pd.Period("2015-01", "M")

    # ---- Precompute ALL neutralized signals (one-time cost) ----
    print(f"\n  Precomputing neutralized signals for {len(candidate_signals)} candidates...")

    # Build projection cache if not provided
    if projection_cache is None:
        from src.gpu.neutralize_batch import ProjectionCache
        print(f"    Building projection matrix cache...")
        projection_cache = ProjectionCache()
        projection_cache.build(
            sector_labels=sector_labels,
            log_mcap=log_mcap,
            beta=beta,
            neutralize_sector=True,
            neutralize_size=True,
            neutralize_beta=True,
        )
        print(f"    Cached {len(projection_cache.projections)} dates")

    precomputed_neutral = {}
    for i, (name, raw) in enumerate(candidate_signals.items()):
        z = standardize_signal(raw, winsorize_pct=0.01)
        if projection_cache._built:
            if GPU_AVAILABLE:
                n = projection_cache.neutralize_fast_gpu(z)
            else:
                n = projection_cache.neutralize_fast(z)
        else:
            n = neutralize_signal(
                z, sector_labels=sector_labels, log_mcap=log_mcap, beta=beta,
                neutralize_sector=True, neutralize_size=True, neutralize_beta=True,
            )
        precomputed_neutral[name] = n
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(candidate_signals)}] neutralized")

    print(f"    All {len(precomputed_neutral)} signals precomputed")

    # ---- Stepwise selection (combine + backtest only, ~1 sec per trial) ----
    remaining = set(candidate_signals.keys())
    selected = []
    history = []
    current_metric = -999.0

    print(f"\n  Forward Stepwise Selection (metric={eval_metric}, max={max_signals})")
    print(f"  Candidates: {len(remaining)}")
    print(f"  {'='*80}")

    for step in range(1, max_signals + 1):
        if not remaining:
            break

        print(f"\n  Step {step}: Testing {len(remaining)} candidates ({_MAX_WORKERS} workers)...",
              flush=True)
        best_name = None
        best_metric = -999.0
        best_result = None

        # Parallel evaluation of all candidates in this step
        candidates_sorted = sorted(remaining)
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS, mp_context=_MP_CONTEXT) as pool:
            futures = {}
            for name in candidates_sorted:
                trial_set = selected + [name]
                fut = pool.submit(
                    _evaluate_precomputed,
                    trial_set, precomputed_neutral, returns, universe,
                    factor_returns, rf, config, oos_start,
                )
                futures[fut] = name

            for fut in as_completed(futures):
                name = futures[fut]
                full_sr, oos_sr, alpha, alpha_t = fut.result()

                trial_metric = {
                    "full_sharpe": full_sr,
                    "oos_sharpe": oos_sr,
                    "ff_alpha_t": alpha_t,
                }.get(eval_metric, full_sr)

                if trial_metric > best_metric:
                    best_metric = trial_metric
                    best_name = name
                    best_result = (full_sr, oos_sr, alpha, alpha_t)

        if best_name is None:
            break

        improvement = best_metric - current_metric if current_metric > -999 else best_metric

        if step > 1 and improvement < min_improvement:
            print(f"\n  Stopping: best improvement {improvement:+.4f} < threshold {min_improvement}")
            break

        selected.append(best_name)
        remaining.discard(best_name)
        current_metric = best_metric
        full_sr, oos_sr, alpha, alpha_t = best_result

        step_result = StepResult(
            step=step,
            added_signal=best_name,
            signal_set=list(selected),
            full_sharpe=full_sr,
            oos_sharpe=oos_sr,
            ff_alpha=alpha,
            ff_alpha_t=alpha_t,
            n_signals=len(selected),
            improvement=improvement,
        )
        history.append(step_result)

        print(f"  + {best_name:35s} → Full SR={full_sr:.4f}  OOS SR={oos_sr:.4f}  "
              f"α={alpha:.4f} (t={alpha_t:.2f})  Δ={improvement:+.4f}")

    # Print summary
    print(f"\n  {'='*80}")
    print(f"  STEPWISE SELECTION COMPLETE")
    print(f"  {'='*80}")
    print(f"\n  {'Step':>5s}  {'Signal':35s} {'Full SR':>8s} {'OOS SR':>8s} {'FF α':>8s} {'α t':>6s} {'Δ':>8s}")
    print(f"  {'-'*85}")
    for r in history:
        print(f"  {r.step:>5d}  {r.added_signal:35s} {r.full_sharpe:>8.4f} {r.oos_sharpe:>8.4f} "
              f"{r.ff_alpha:>8.4f} {r.ff_alpha_t:>6.2f} {r.improvement:>+8.4f}")

    if history:
        best_step = max(history, key=lambda r: getattr(r, eval_metric.replace("ff_", "")))
        print(f"\n  Best portfolio ({eval_metric}): {best_step.n_signals} signals")
        print(f"    Signals: {best_step.signal_set}")
        print(f"    Full Sharpe: {best_step.full_sharpe:.4f}")
        print(f"    OOS Sharpe:  {best_step.oos_sharpe:.4f}")
        print(f"    FF Alpha:    {best_step.ff_alpha:.4f} (t={best_step.ff_alpha_t:.2f})")

    return history


def _evaluate_precomputed(
    signal_names: list[str],
    precomputed_neutral: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    factor_returns: pd.DataFrame,
    rf: pd.Series,
    config: PipelineConfig,
    oos_start: pd.Period,
    backtest_start: str = None,
    backtest_end: str = None,
) -> tuple[float, float, float, float]:
    """Combine precomputed neutralized signals + backtest. ~1 sec per call.

    Args:
        backtest_start: override config.dates.burn_in_end (for inner folds)
        backtest_end: override config.dates.end (for inner folds)
    """

    # Look up precomputed neutralized signals
    neutral = {name: precomputed_neutral[name] for name in signal_names}

    # Combine (equal weight)
    composite = combine_signals(neutral, method="equal")

    # Backtest (naive L/S)
    bt = WalkForwardBacktest(
        config=config,
        alpha_scores=composite,
        stock_returns=returns,
        factor_returns=factor_returns,
        beta=pd.DataFrame(),  # beta not needed for naive L/S
        sector_labels=pd.DataFrame(),
        universe=universe,
        rf=rf,
        use_optimizer=False,
    )
    start = backtest_start or config.dates.burn_in_end
    end = backtest_end or config.dates.end
    result = bt.run(start_date=start, end_date=end)
    net = result.net_returns

    if len(net) < 24:
        return 0, 0, 0, 0

    # Selection metric: use ONLY pre-OOS period (exclude 2015-2019 from selection)
    # This prevents look-ahead bias — OOS is truly unseen
    pre_oos_net = net[net.index < oos_start]
    oos_net = net[net.index >= oos_start]

    selection_sr = sharpe_ratio(pre_oos_net) if len(pre_oos_net) > 12 else 0
    oos_sr = sharpe_ratio(oos_net) if len(oos_net) > 12 else 0

    attr = factor_attribution(net, factor_returns, rf)
    alpha = attr.get("alpha", 0)
    alpha_t = attr.get("alpha_t_stat", 0)

    return selection_sr, oos_sr, alpha, alpha_t


def _evaluate_fold(
    signal_names: list[str],
    precomputed_neutral: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    config: PipelineConfig,
    fold_start: str,
    fold_end: str,
    methods: tuple[str, ...] = ("equal", "ic_weighted", "inverse_vol"),
    lookback: int = 36,
) -> dict:
    """Evaluate a signal set on a single fold using multiple combination methods.

    Returns averaged metrics across all methods (method-agnostic selection).
    """
    from src.analytics.performance import sharpe_ratio as sr, max_drawdown as mdd

    neutral = {name: precomputed_neutral[name] for name in signal_names}

    sharpes = []
    turnovers = []
    max_dds = []
    per_method = {}

    for method in methods:
        composite = combine_signals(
            neutral, method=method,
            returns=returns, universe=universe,
            lookback=lookback,
        )

        bt = WalkForwardBacktest(
            config=config,
            alpha_scores=composite,
            stock_returns=returns,
            factor_returns=pd.DataFrame(),
            beta=pd.DataFrame(),
            sector_labels=pd.DataFrame(),
            universe=universe,
            rf=pd.Series(dtype=float),
            use_optimizer=False,
        )
        result = bt.run(start_date=fold_start, end_date=fold_end)
        net = result.net_returns

        if len(net) < 12:
            sharpes.append(0)
            turnovers.append(1.0)
            max_dds.append(1.0)
        else:
            sharpes.append(sr(net))
            turnovers.append(float(result.turnover.mean()) if len(result.turnover) > 0 else 0)
            max_dds.append(mdd(net))

        per_method[method] = sharpes[-1]

    return {
        "sharpe": float(np.mean(sharpes)),
        "turnover": float(np.mean(turnovers)),
        "max_drawdown": float(np.mean(max_dds)),
        "per_method_sharpes": per_method,
    }


# ===========================================================================
# Nested multi-fold stepwise selection
# ===========================================================================

@dataclass
class NestedStepResult:
    step: int
    added_signal: str
    signal_set: list[str]
    fold_sharpes: list[float]
    mean_sharpe: float
    median_sharpe: float
    penalized_score: float
    fold_turnovers: list[float]
    fold_max_drawdowns: list[float]
    stability: float
    improvement: float


def _penalized_score(
    fold_sharpes: list[float],
    fold_turnovers: list[float],
    stability: float,
    concentration: float,
    lambda_turnover: float,
    lambda_instability: float,
    lambda_concentration: float,
) -> float:
    """Multi-criteria scoring: penalize turnover, instability, concentration."""
    mean_sharpe = np.mean(fold_sharpes)
    mean_turnover = np.mean(fold_turnovers)
    instability = 1.0 - stability
    return (
        mean_sharpe
        - lambda_turnover * mean_turnover
        - lambda_instability * instability
        - lambda_concentration * concentration
    )


def _max_corr_with_set(
    candidate: str,
    existing: list[str],
    precomputed_neutral: dict[str, pd.DataFrame],
    n_sample_dates: int = 60,
) -> float:
    """Max pairwise correlation of candidate with existing signals."""
    if not existing:
        return 0.0

    cand_sig = precomputed_neutral[candidate]
    sample_dates = cand_sig.dropna(how="all").index
    if len(sample_dates) > n_sample_dates:
        rng = np.random.default_rng(42)
        sample_dates = rng.choice(sample_dates, size=n_sample_dates, replace=False)

    max_corr = 0.0
    for ex_name in existing:
        ex_sig = precomputed_neutral[ex_name]
        corrs = []
        for t in sample_dates:
            if t not in ex_sig.index:
                continue
            s1 = cand_sig.loc[t].dropna()
            s2 = ex_sig.loc[t].reindex(s1.index).dropna()
            common = s1.index.intersection(s2.index)
            if len(common) > 20:
                corrs.append(abs(np.corrcoef(s1.loc[common], s2.loc[common])[0, 1]))
        if corrs:
            max_corr = max(max_corr, np.mean(corrs))

    return max_corr


def _evaluate_candidate_nested(
    name: str,
    trial_set: list[str],
    precomputed_neutral: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    config: PipelineConfig,
    fold_defs: list,
    inner_start: str,
    selected: list[str],
    current_fold_sharpes: list[float] | None,
    lambda_turnover: float,
    lambda_instability: float,
    lambda_concentration: float,
) -> dict:
    """Evaluate one candidate across all folds. Top-level function for pickling."""
    fold_metrics = []
    for fd in fold_defs:
        metrics = _evaluate_fold(
            trial_set, precomputed_neutral, returns, universe, config,
            fold_start=inner_start, fold_end=fd.val_end,
        )
        fold_metrics.append(metrics)

    fold_sharpes = [m["sharpe"] for m in fold_metrics]
    fold_turnovers = [m["turnover"] for m in fold_metrics]
    fold_drawdowns = [m["max_drawdown"] for m in fold_metrics]

    if current_fold_sharpes is not None:
        improvements = [
            fs - cs for fs, cs in zip(fold_sharpes, current_fold_sharpes)
        ]
        stability = sum(1 for imp in improvements if imp > 0) / len(fold_defs)
    else:
        stability = sum(1 for fs in fold_sharpes if fs > 0) / len(fold_defs)

    concentration = _max_corr_with_set(name, selected, precomputed_neutral)

    score = _penalized_score(
        fold_sharpes, fold_turnovers, stability, concentration,
        lambda_turnover, lambda_instability, lambda_concentration,
    )

    return {
        "fold_sharpes": fold_sharpes,
        "fold_turnovers": fold_turnovers,
        "fold_drawdowns": fold_drawdowns,
        "stability": stability,
        "concentration": concentration,
        "score": score,
    }


def forward_stepwise_nested(
    stable_candidates: list[str],
    precomputed_neutral: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    config: PipelineConfig,
    fold_defs: list,
    validation_config=None,
    max_signals: int = 15,
) -> list[NestedStepResult]:
    """Forward stepwise with cross-fold penalized scoring.

    For each candidate addition, evaluates across ALL inner folds.
    Uses penalized score (Sharpe - turnover - instability - concentration).
    Stops when multi-criteria requirements aren't met.
    """
    if validation_config is None:
        from src.config import NestedValidationConfig
        validation_config = NestedValidationConfig()

    vc = validation_config
    inner_start = vc.inner_start

    remaining = set(stable_candidates)
    selected = []
    history = []
    current_score = -999.0
    current_fold_sharpes = [0.0] * len(fold_defs)
    current_fold_turnovers = [0.0] * len(fold_defs)
    current_fold_drawdowns = [0.0] * len(fold_defs)

    print(f"\n  Nested Forward Stepwise Selection")
    print(f"  Candidates: {len(remaining)}, Folds: {len(fold_defs)}")
    print(f"  Penalties: λ_TO={vc.lambda_turnover}, λ_inst={vc.lambda_instability}, λ_conc={vc.lambda_concentration}")
    print(f"  {'='*90}")

    for step in range(1, max_signals + 1):
        if not remaining:
            break

        print(f"\n  Step {step}: Testing {len(remaining)} candidates × {len(fold_defs)} folds × 3 methods ({_MAX_WORKERS} workers)...",
              flush=True)
        best_name = None
        best_score = -999.0
        best_result = None

        # Parallel evaluation of all candidates in this step
        candidates_sorted = sorted(remaining)
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS, mp_context=_MP_CONTEXT) as pool:
            futures = {}
            for name in candidates_sorted:
                trial_set = selected + [name]
                fut = pool.submit(
                    _evaluate_candidate_nested,
                    name, trial_set, precomputed_neutral, returns, universe, config,
                    fold_defs, inner_start, selected,
                    current_fold_sharpes if step > 1 else None,
                    vc.lambda_turnover, vc.lambda_instability, vc.lambda_concentration,
                )
                futures[fut] = name

            for fut in as_completed(futures):
                name = futures[fut]
                cand_result = fut.result()

                if cand_result["score"] > best_score:
                    best_score = cand_result["score"]
                    best_name = name
                    best_result = cand_result

        if best_name is None:
            break

        br = best_result
        improvement = best_score - current_score if current_score > -999 else best_score

        # Multi-criteria stopping checks
        if step > 1:
            mean_sharpe_imp = np.mean(br["fold_sharpes"]) - np.mean(current_fold_sharpes)
            median_sharpe_imp = np.median(br["fold_sharpes"]) - np.median(current_fold_sharpes)

            if mean_sharpe_imp < vc.min_improvement_mean:
                print(f"\n  Stopping: mean Sharpe improvement {mean_sharpe_imp:+.4f} < {vc.min_improvement_mean}")
                break
            if median_sharpe_imp < vc.min_improvement_median:
                print(f"\n  Stopping: median Sharpe improvement {median_sharpe_imp:+.4f} < {vc.min_improvement_median}")
                break

            # Check turnover and drawdown degradation per fold
            to_deltas = [
                ft - ct for ft, ct in zip(br["fold_turnovers"], current_fold_turnovers)
            ]
            dd_deltas = [
                fd - cd for fd, cd in zip(br["fold_drawdowns"], current_fold_drawdowns)
            ]
            if any(d > vc.max_turnover_delta for d in to_deltas):
                print(f"\n  Stopping: turnover degradation exceeds {vc.max_turnover_delta}")
                break
            if any(d > vc.max_drawdown_delta for d in dd_deltas):
                print(f"\n  Stopping: drawdown degradation exceeds {vc.max_drawdown_delta}")
                break

        selected.append(best_name)
        remaining.discard(best_name)
        current_score = best_score
        current_fold_sharpes = br["fold_sharpes"]
        current_fold_turnovers = br["fold_turnovers"]
        current_fold_drawdowns = br["fold_drawdowns"]

        step_result = NestedStepResult(
            step=step,
            added_signal=best_name,
            signal_set=list(selected),
            fold_sharpes=br["fold_sharpes"],
            mean_sharpe=float(np.mean(br["fold_sharpes"])),
            median_sharpe=float(np.median(br["fold_sharpes"])),
            penalized_score=best_score,
            fold_turnovers=br["fold_turnovers"],
            fold_max_drawdowns=br["fold_drawdowns"],
            stability=br["stability"],
            improvement=improvement,
        )
        history.append(step_result)

        print(f"  + {best_name:35s} → mean SR={step_result.mean_sharpe:.4f}  "
              f"med SR={step_result.median_sharpe:.4f}  score={best_score:.4f}  "
              f"stab={br['stability']:.2f}  Δ={improvement:+.4f}")

    # Print summary
    print(f"\n  {'='*90}")
    print(f"  NESTED STEPWISE COMPLETE")
    print(f"  {'='*90}")
    if history:
        print(f"\n  {'Step':>5s}  {'Signal':35s} {'Mean SR':>8s} {'Med SR':>8s} {'Score':>8s} {'Stab':>6s} {'Δ':>8s}")
        print(f"  {'-'*85}")
        for r in history:
            print(f"  {r.step:>5d}  {r.added_signal:35s} {r.mean_sharpe:>8.4f} {r.median_sharpe:>8.4f} "
                  f"{r.penalized_score:>8.4f} {r.stability:>6.2f} {r.improvement:>+8.4f}")

        best = history[-1]
        print(f"\n  Final portfolio: {best.step} signals")
        print(f"    Signals: {best.signal_set}")
        print(f"    Mean Sharpe:   {best.mean_sharpe:.4f}")
        print(f"    Median Sharpe: {best.median_sharpe:.4f}")
        print(f"    Per-fold Sharpes: {[f'{s:.4f}' for s in best.fold_sharpes]}")

    return history
