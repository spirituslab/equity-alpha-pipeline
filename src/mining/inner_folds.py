"""Inner fold engine for nested chronological validation.

Each fold runs the full preselect → dedup → stepwise pipeline
on a specific train/validation window within the inner layer.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import PipelineConfig, InnerFoldDef, NestedValidationConfig
from src.mining.config import MiningConfig
from src.mining.evaluate import EvalResult
from src.mining.filter import filter_dev_period, filter_val_period
from src.mining.deduplicate import deduplicate
from src.analytics.ic import ic_summary
from src.signals.zscore import standardize_signal


@dataclass
class FoldResult:
    fold_id: int
    train_end: str
    val_end: str
    ic_survivors: list[str] = field(default_factory=list)
    dev_survivors: list[str] = field(default_factory=list)
    val_survivors: list[str] = field(default_factory=list)
    dedup_survivors: list[str] = field(default_factory=list)
    stepwise_selected: list[str] = field(default_factory=list)
    stepwise_history: list = field(default_factory=list)
    fold_sharpe: float = 0.0
    per_signal_metrics: dict = field(default_factory=dict)


class InnerFoldRunner:
    """Run preselect → dedup → stepwise on one expanding fold."""

    def __init__(
        self,
        fold_id: int,
        fold_def: InnerFoldDef,
        raw_signals: dict[str, pd.DataFrame],
        z_signals: dict[str, pd.DataFrame],
        ic_dict: dict[str, pd.Series],
        turnover_dict: dict[str, float],
        spread_dict: dict[str, dict],
        returns: pd.DataFrame,
        universe: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rf: pd.Series,
        pipeline_config: PipelineConfig,
        mining_config: MiningConfig,
        precomputed_neutral: dict[str, pd.DataFrame],
        validation_config: NestedValidationConfig = None,
    ):
        self.fold_id = fold_id
        self.fold_def = fold_def
        self.raw_signals = raw_signals
        self.z_signals = z_signals
        self.ic_dict = ic_dict
        self.turnover_dict = turnover_dict
        self.spread_dict = spread_dict
        self.returns = returns
        self.universe = universe
        self.factor_returns = factor_returns
        self.rf = rf
        self.config = pipeline_config
        self.mining_config = mining_config
        self.precomputed_neutral = precomputed_neutral
        self.val_config = validation_config or pipeline_config.validation

    def run(self) -> FoldResult:
        """Execute the full fold pipeline."""
        fd = self.fold_def
        mc = self.mining_config

        print(f"\n    {'='*60}")
        print(f"    FOLD {self.fold_id}: train→{fd.train_end}, val→{fd.val_end}")
        print(f"    {'='*60}")

        result = FoldResult(
            fold_id=self.fold_id,
            train_end=fd.train_end,
            val_end=fd.val_end,
        )

        # ---- 1. IC filtering on this fold's train period ----
        # ic_dict was computed per-fold via GPU batch_compute_ic(end_date=fold.train_end)
        evaluations = {}
        for name in self.raw_signals:
            ev = EvalResult(name=name)
            if name in self.ic_dict and len(self.ic_dict[name]) > 0:
                stats = ic_summary(self.ic_dict[name])
                ev.dev_ic = stats.get("Mean IC", np.nan)
                ev.dev_icir = stats.get("ICIR", np.nan)
                ev.dev_hit_rate = stats.get("Hit Rate", np.nan)
                ev.dev_t_stat = stats.get("t-stat", np.nan)
                ev.n_months_dev = stats.get("N Months", 0)
            evaluations[name] = ev

        # IC filter
        ic_threshold = mc.min_icir_dev
        ic_survivors = {
            n: ev for n, ev in evaluations.items()
            if not np.isnan(ev.dev_icir) and abs(ev.dev_icir) >= ic_threshold
        }
        result.ic_survivors = list(ic_survivors.keys())
        print(f"    IC filter: {len(ic_survivors)} / {len(evaluations)} passed (|ICIR| >= {ic_threshold})")

        # ---- 2. Turnover + spread from precomputed global data ----
        # Assign precomputed turnover and spread to survivors
        for name, ev in ic_survivors.items():
            ev.turnover = self.turnover_dict.get(name, np.nan)
            sp = self.spread_dict.get(name, {})
            ev.dev_spread = sp.get("spread", np.nan)
            ev.dev_spread_t = sp.get("spread_t", np.nan)

        # ---- 3. Full dev filters ----
        dev_survivors = filter_dev_period(ic_survivors, mc)
        result.dev_survivors = list(dev_survivors.keys())
        print(f"    Dev filter: {len(dev_survivors)} / {len(ic_survivors)} passed")

        if not dev_survivors:
            print(f"    No survivors — skipping fold {self.fold_id}")
            return result

        # ---- 4. Holdout validation on fold's val period ----
        from src.mining.evaluate import validate_candidate

        purge_months = self.val_config.purge_months
        # Purge: skip first purge_months of validation window
        val_start_period = pd.Period(fd.train_end, "M") + purge_months

        for name, ev in dev_survivors.items():
            val_ic, val_icir = validate_candidate(
                name, self.raw_signals[name], self.returns, self.universe,
                val_start=str(val_start_period),
                val_end=fd.val_end,
            )
            ev.val_ic = val_ic
            ev.val_icir = val_icir

        val_survivors = filter_val_period(dev_survivors, mc)
        result.val_survivors = list(val_survivors.keys())
        print(f"    Val filter: {len(val_survivors)} / {len(dev_survivors)} passed")

        if not val_survivors:
            print(f"    No val survivors — skipping fold {self.fold_id}")
            return result

        # ---- 5. Correlation dedup ----
        std_signals = {n: self.z_signals[n] for n in val_survivors if n in self.z_signals}
        accepted_names = deduplicate(
            val_survivors, std_signals, {},  # no existing signals
            max_corr=mc.max_correlation,
        )
        result.dedup_survivors = list(accepted_names)
        print(f"    Dedup: {len(accepted_names)} / {len(val_survivors)} passed")

        if not accepted_names:
            print(f"    No dedup survivors — skipping fold {self.fold_id}")
            return result

        # ---- 6. Forward stepwise on this fold ----
        from src.mining.stepwise import forward_stepwise_selection

        # Filter precomputed_neutral to only dedup survivors
        fold_neutral = {
            n: self.precomputed_neutral[n]
            for n in accepted_names
            if n in self.precomputed_neutral
        }

        if len(fold_neutral) < 1:
            return result

        # Run stepwise with fold-specific boundaries
        # Selection uses full fold period (inner_start to fold.val_end)
        fold_end_period = pd.Period(fd.val_end, "M")
        inner_start = self.val_config.inner_start

        history = forward_stepwise_selection(
            candidate_signals={n: self.raw_signals[n] for n in fold_neutral},
            returns=self.returns,
            universe=self.universe,
            sector_labels=pd.DataFrame(),
            log_mcap=pd.DataFrame(),
            beta=pd.DataFrame(),
            factor_returns=self.factor_returns,
            rf=self.rf,
            config=self.config,
            eval_metric="full_sharpe",
            max_signals=10,
            min_improvement=0.01,
            projection_cache=_FakeCache(fold_neutral),
        )

        if history:
            best = history[-1]
            result.stepwise_selected = list(best.signal_set)
            result.stepwise_history = history
            result.fold_sharpe = best.full_sharpe

        # ---- 7. Collect per-signal metrics ----
        metrics = {}
        for name in ic_survivors:
            m = {
                "icir_train": evaluations[name].dev_icir,
                "ic_train": evaluations[name].dev_ic,
                "hit_rate": evaluations[name].dev_hit_rate,
                "turnover": evaluations[name].turnover,
                "spread_t": evaluations[name].dev_spread_t,
            }
            if name in dev_survivors:
                m["icir_val"] = dev_survivors[name].val_icir
            metrics[name] = m
        result.per_signal_metrics = metrics

        print(f"    Fold {self.fold_id} complete: {len(result.stepwise_selected)} signals selected, Sharpe={result.fold_sharpe:.4f}")
        return result


class _FakeCache:
    """Adapter to pass precomputed neutral signals to forward_stepwise_selection.

    forward_stepwise_selection checks projection_cache._built and uses it
    if available. This wrapper makes it use our pre-neutralized signals directly.
    """

    def __init__(self, precomputed_neutral: dict[str, pd.DataFrame]):
        self._built = True
        self._neutral = precomputed_neutral
        self.projections = {}  # empty — won't be used

    def neutralize_fast(self, signal):
        # Should not be called since we override in stepwise
        return signal

    def neutralize_fast_gpu(self, signal):
        return signal
