"""Middle-layer model comparison for nested validation.

After inner-layer selection freezes the signal set, this module
evaluates different combination and construction methods on the
middle period (2010-2014) to select the winner.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.config import PipelineConfig
from src.signals.combine import combine_signals
from src.portfolio.backtest import WalkForwardBacktest
from src.analytics.performance import sharpe_ratio, sortino_ratio, max_drawdown
from src.analytics.statistical_tests import diebold_mariano_test


@dataclass
class ModelComparisonResult:
    method_results: dict[str, dict] = field(default_factory=dict)
    method_sharpes: dict[str, float] = field(default_factory=dict)
    selected_method: str = "equal"
    dm_tests: dict[tuple[str, str], dict] = field(default_factory=dict)
    freeze_manifest: dict = field(default_factory=dict)


class ModelComparison:
    """Compare combination methods on the middle validation layer."""

    def __init__(
        self,
        frozen_signals: list[str],
        precomputed_neutral: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        universe: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rf: pd.Series,
        config: PipelineConfig,
    ):
        self.frozen_signals = frozen_signals
        self.precomputed_neutral = precomputed_neutral
        self.returns = returns
        self.universe = universe
        self.factor_returns = factor_returns
        self.rf = rf
        self.config = config
        self.vc = config.validation

    def run(self) -> ModelComparisonResult:
        """Evaluate EW, IC-weighted, inverse-vol on middle period."""
        vc = self.vc
        middle_start = vc.middle_start
        middle_end = vc.middle_end

        # Warm-up: start backtest 60 months before middle period
        warmup_start = str(pd.Period(middle_start, "M") - self.config.backtest.lookback_cov)

        neutral = {n: self.precomputed_neutral[n] for n in self.frozen_signals
                   if n in self.precomputed_neutral}

        print(f"\n  {'='*70}")
        print(f"  MODEL COMPARISON: {len(neutral)} signals on {middle_start} to {middle_end}")
        print(f"  {'='*70}")

        methods = {
            "equal": lambda: combine_signals(neutral, method="equal"),
            "ic_weighted": lambda: combine_signals(
                neutral, method="ic_weighted",
                returns=self.returns, universe=self.universe,
                lookback=self.config.backtest.lookback_signal,
            ),
            "inverse_vol": lambda: combine_signals(
                neutral, method="inverse_vol",
                returns=self.returns, universe=self.universe,
                lookback=self.config.backtest.lookback_signal,
            ),
        }

        middle_start_period = pd.Period(middle_start, "M")
        middle_end_period = pd.Period(middle_end, "M")

        net_returns = {}
        method_metrics = {}

        for method_name, combine_fn in methods.items():
            print(f"\n    {method_name}:")
            composite = combine_fn()

            bt = WalkForwardBacktest(
                config=self.config,
                alpha_scores=composite,
                stock_returns=self.returns,
                factor_returns=self.factor_returns,
                beta=pd.DataFrame(),
                sector_labels=pd.DataFrame(),
                universe=self.universe,
                rf=self.rf,
                use_optimizer=False,
            )
            result = bt.run(start_date=warmup_start, end_date=middle_end)
            net = result.net_returns

            # Only measure metrics on middle period
            middle_net = net[(net.index >= middle_start_period) & (net.index <= middle_end_period)]

            if len(middle_net) < 12:
                print(f"      Insufficient data ({len(middle_net)} months)")
                method_metrics[method_name] = {
                    "sharpe": 0, "sortino": 0, "max_drawdown": 0,
                    "mean_return": 0, "n_months": len(middle_net),
                }
                net_returns[method_name] = middle_net
                continue

            sr = sharpe_ratio(middle_net)
            metrics = {
                "sharpe": sr,
                "sortino": sortino_ratio(middle_net),
                "max_drawdown": max_drawdown(middle_net),
                "mean_return": float(middle_net.mean() * 12),
                "turnover": float(result.turnover.mean()) if len(result.turnover) > 0 else 0,
                "n_months": len(middle_net),
            }
            method_metrics[method_name] = metrics
            net_returns[method_name] = middle_net

            print(f"      Sharpe: {sr:.4f}  Sortino: {metrics['sortino']:.4f}  "
                  f"MaxDD: {metrics['max_drawdown']:.4f}  TO: {metrics['turnover']:.4f}")

        # Pairwise Diebold-Mariano tests
        dm_tests = {}
        method_names = list(net_returns.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                if len(net_returns[m1]) > 12 and len(net_returns[m2]) > 12:
                    dm = diebold_mariano_test(net_returns[m1], net_returns[m2])
                    dm_tests[(m1, m2)] = dm
                    sig = "*" if dm["Significant (5%)"] else ""
                    print(f"\n    DM test {m1} vs {m2}: stat={dm['DM Statistic']:.3f} p={dm['p-value']:.4f} {sig}")

        # Select winner: highest Sharpe, but default to EW if no significant difference
        method_sharpes = {m: met["sharpe"] for m, met in method_metrics.items()}
        best_method = max(method_sharpes, key=method_sharpes.get)

        # If best is not significantly better than EW, default to EW (most robust)
        if best_method != "equal":
            key = ("equal", best_method) if ("equal", best_method) in dm_tests else (best_method, "equal")
            if key in dm_tests:
                if not dm_tests[key]["Significant (5%)"]:
                    print(f"\n    {best_method} not significantly better than equal — defaulting to equal")
                    best_method = "equal"

        print(f"\n    Selected method: {best_method} (Sharpe={method_sharpes.get(best_method, 0):.4f})")

        return ModelComparisonResult(
            method_results=method_metrics,
            method_sharpes=method_sharpes,
            selected_method=best_method,
            dm_tests=dm_tests,
        )
