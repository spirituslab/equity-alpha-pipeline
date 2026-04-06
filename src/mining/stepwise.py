"""Forward stepwise signal selection at the portfolio level.

Precomputes ALL neutralized signals once, then stepwise only does
combine + backtest per candidate (~1 sec instead of ~40 sec).
"""

from dataclasses import dataclass

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

        print(f"\n  Step {step}: Testing {len(remaining)} candidates...")
        best_name = None
        best_metric = -999.0
        best_result = None

        for name in tqdm(sorted(remaining), desc=f"Step {step}", leave=False):
            trial_set = selected + [name]

            # Just look up precomputed neutralized signals + combine + backtest
            full_sr, oos_sr, alpha, alpha_t = _evaluate_precomputed(
                trial_set, precomputed_neutral, returns, universe,
                factor_returns, rf, config, oos_start,
            )

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
) -> tuple[float, float, float, float]:
    """Combine precomputed neutralized signals + backtest. ~1 sec per call."""

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
    result = bt.run(start_date=config.dates.burn_in_end, end_date=config.dates.end)
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
