"""Stock-level walk-forward backtest engine.

At each monthly rebalance date t:
1. Receive composite alpha scores (pre-computed)
2. Estimate factor model covariance from trailing window
3. Optimize portfolio with neutrality constraints
4. Record holdings, compute realized t+1 return
5. Deduct transaction costs
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig
from src.portfolio.risk_model import FactorRiskModel
from src.portfolio.optimization import long_short_optimize
from src.portfolio.construction import signal_to_target_weights


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    gross_returns: pd.Series = None
    net_returns: pd.Series = None
    holdings: dict = field(default_factory=dict)      # date -> pd.Series of weights
    turnover: pd.Series = None
    costs: pd.Series = None
    realized_beta: pd.Series = None
    realized_sector_net: pd.DataFrame = None
    n_long: pd.Series = None
    n_short: pd.Series = None


class WalkForwardBacktest:
    """Walk-forward backtest with stock-level neutrality constraints.

    Usage:
        bt = WalkForwardBacktest(config, alpha, returns, factor_returns,
                                  beta, sector_labels, universe)
        result = bt.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        alpha_scores: pd.DataFrame,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        beta: pd.DataFrame,
        sector_labels: pd.DataFrame,
        universe: pd.DataFrame,
        rf: pd.Series = None,
        use_optimizer: bool = True,
    ):
        self.config = config
        self.alpha = alpha_scores
        self.returns = stock_returns
        self.factor_returns = factor_returns
        self.beta = beta
        self.sector_labels = sector_labels
        self.universe = universe
        self.rf = rf
        self.use_optimizer = use_optimizer

    def run(self, start_date: str = None, end_date: str = None) -> BacktestResult:
        """Execute walk-forward backtest."""
        opt = self.config.optimization
        bt = self.config.backtest
        cost_bps = self.config.costs.base_bps

        # Determine rebalance dates
        alpha_dates = self.alpha.dropna(how="all").index
        if start_date:
            alpha_dates = alpha_dates[alpha_dates >= pd.Period(start_date, "M")]
        if end_date:
            alpha_dates = alpha_dates[alpha_dates <= pd.Period(end_date, "M")]

        # Need at least lookback_cov months of data before first rebalance
        min_date_idx = bt.lookback_cov
        if min_date_idx >= len(self.returns.index):
            raise ValueError("Not enough data for covariance lookback")

        earliest = self.returns.index[min_date_idx]
        alpha_dates = alpha_dates[alpha_dates >= earliest]

        result = BacktestResult()
        gross_rets = {}
        net_rets = {}
        turnover_series = {}
        cost_series = {}
        beta_series = {}
        n_long_series = {}
        n_short_series = {}

        w_prev = None

        for t in tqdm(alpha_dates, desc="Backtesting"):
            t_plus_1 = t + 1
            if t_plus_1 not in self.returns.index:
                continue

            # Get universe at t
            if t not in self.universe.index:
                continue
            univ = self.universe.loc[t]
            stocks = univ[univ].index.tolist()

            if len(stocks) < 50:
                continue

            # Get alpha scores at t
            alpha_t = self.alpha.loc[t].reindex(stocks).dropna()
            if len(alpha_t) < 50:
                continue

            active_stocks = alpha_t.index.tolist()

            if self.use_optimizer:
                # Estimate covariance from trailing window
                t_idx = self.returns.index.get_loc(t)
                start_idx = max(0, t_idx - bt.lookback_cov)
                trailing_returns = self.returns.iloc[start_idx:t_idx][active_stocks].dropna(axis=1, how="all")
                trailing_stocks = trailing_returns.columns.tolist()

                # Re-align alpha to trailing stocks
                alpha_t = alpha_t.reindex(trailing_stocks).dropna()
                active_stocks = alpha_t.index.tolist()

                if len(active_stocks) < 50:
                    continue

                # Factor risk model
                risk_model = FactorRiskModel()
                trailing_factor_ret = self.factor_returns.iloc[start_idx:t_idx]
                rf_trailing = self.rf.iloc[start_idx:t_idx] if self.rf is not None else None
                risk_model.fit(
                    trailing_returns[active_stocks],
                    trailing_factor_ret,
                    rf=rf_trailing,
                )

                sigma = risk_model.get_covariance(active_stocks)
                beta_vec = risk_model.get_beta(active_stocks)

                # Sector labels as integers
                if t in self.sector_labels.index:
                    sec = self.sector_labels.loc[t].reindex(active_stocks)
                    unique_sec = sec.dropna().unique()
                    sec_map = {s: i for i, s in enumerate(sorted(unique_sec))}
                    sector_int = sec.map(sec_map).fillna(-1).values.astype(float)
                else:
                    sector_int = np.zeros(len(active_stocks))

                # Previous weights aligned to current stocks
                w_prev_aligned = np.zeros(len(active_stocks))
                if w_prev is not None:
                    for i, s in enumerate(active_stocks):
                        if s in w_prev.index:
                            w_prev_aligned[i] = w_prev[s]

                # Optimize
                w = long_short_optimize(
                    alpha=alpha_t.values,
                    sigma=sigma,
                    beta=beta_vec,
                    sector=sector_int,
                    w_prev=w_prev_aligned,
                    risk_aversion=opt.risk_aversion,
                    turnover_penalty=opt.turnover_penalty,
                    max_stock_weight=opt.max_stock_weight,
                    max_sector_net=opt.max_sector_net,
                    max_gross_leverage=opt.max_gross_leverage,
                    max_beta_exposure=opt.max_beta_exposure,
                )

                weights = pd.Series(w, index=active_stocks)
            else:
                # Naive long/short
                weights = signal_to_target_weights(
                    alpha_t, n_long=bt.n_long, n_short=bt.n_short,
                    weight_method="equal_weight",
                )
                weights = weights[weights != 0]
                active_stocks = weights.index.tolist()

            # Store holdings
            result.holdings[t] = weights

            # Compute realized return at t+1
            ret_t1 = self.returns.loc[t_plus_1].reindex(active_stocks).fillna(0)
            portfolio_return = (weights * ret_t1).sum()
            gross_rets[t_plus_1] = portfolio_return

            # Transaction costs
            if w_prev is not None:
                w_current = weights.reindex(w_prev.index.union(weights.index)).fillna(0)
                w_old = w_prev.reindex(w_current.index).fillna(0)
                two_way_turnover = (w_current - w_old).abs().sum()
            else:
                two_way_turnover = weights.abs().sum() * 2  # initial build

            tc = cost_bps / 10000.0 * two_way_turnover / 2  # one-way cost on half the turnover
            net_rets[t_plus_1] = portfolio_return - tc
            turnover_series[t_plus_1] = two_way_turnover
            cost_series[t_plus_1] = tc

            # Track realized beta
            if t in self.beta.index:
                b = self.beta.loc[t].reindex(active_stocks).fillna(1.0)
                beta_series[t_plus_1] = (weights * b).sum()

            # Track positions
            n_long_series[t_plus_1] = (weights > 0.001).sum()
            n_short_series[t_plus_1] = (weights < -0.001).sum()

            w_prev = weights

        result.gross_returns = pd.Series(gross_rets, dtype=float)
        result.net_returns = pd.Series(net_rets, dtype=float)
        result.turnover = pd.Series(turnover_series, dtype=float)
        result.costs = pd.Series(cost_series, dtype=float)
        result.realized_beta = pd.Series(beta_series, dtype=float)
        result.n_long = pd.Series(n_long_series, dtype=float)
        result.n_short = pd.Series(n_short_series, dtype=float)

        return result
