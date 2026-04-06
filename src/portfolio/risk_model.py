"""Factor model covariance estimation.

Sigma_stock = B @ Sigma_factor @ B' + D

Reduces the N x N estimation problem (N ~ 500 stocks) to a K x K problem
(K ~ 6 factors), making covariance estimation feasible with limited
time-series observations.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class FactorRiskModel:
    """Statistical factor risk model using Fama-French factors.

    For each stock, estimates factor loadings via time-series regression:
        r_{i,t} - rf_t = alpha_i + B_i @ f_t + epsilon_{i,t}

    Then constructs covariance as:
        Sigma = B @ F @ B' + D

    where F = factor covariance (Ledoit-Wolf), D = diagonal idiosyncratic.
    """

    def __init__(self):
        self.loadings_ = None
        self.factor_cov_ = None
        self.idio_var_ = None
        self.fitted_stocks_ = None

    def fit(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rf: pd.Series = None,
        min_obs: int = 24,
    ) -> "FactorRiskModel":
        """Estimate factor loadings and idiosyncratic risk.

        Args:
            stock_returns: (T x N) monthly stock returns (decimal)
            factor_returns: (T x K) Fama-French factor returns (decimal)
            rf: (T,) risk-free rate. If None, uses raw returns.
            min_obs: minimum observations for regression
        """
        # Align dates
        common_dates = stock_returns.index.intersection(factor_returns.index)
        if rf is not None:
            common_dates = common_dates.intersection(rf.index)

        stock_ret = stock_returns.loc[common_dates]
        factor_ret = factor_returns.loc[common_dates]

        if rf is not None:
            rf_vals = rf.loc[common_dates]
            # Subtract risk-free from stock returns
            stock_ret = stock_ret.sub(rf_vals, axis=0)

        factor_names = factor_ret.columns.tolist()
        n_factors = len(factor_names)

        self.loadings_ = pd.DataFrame(
            np.nan, index=stock_ret.columns, columns=factor_names
        )
        self.idio_var_ = pd.Series(np.nan, index=stock_ret.columns)

        X = factor_ret.values
        X_with_const = np.column_stack([np.ones(len(X)), X])

        for stock in stock_ret.columns:
            y = stock_ret[stock].values
            valid = ~np.isnan(y)
            if valid.sum() < min_obs:
                continue

            y_v = y[valid]
            X_v = X_with_const[valid]

            try:
                beta_hat, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
                self.loadings_.loc[stock] = beta_hat[1:]  # skip intercept
                residuals = y_v - X_v @ beta_hat
                self.idio_var_[stock] = np.var(residuals, ddof=1)
            except np.linalg.LinAlgError:
                continue

        # Factor covariance via Ledoit-Wolf
        lw = LedoitWolf().fit(factor_ret.dropna())
        self.factor_cov_ = pd.DataFrame(
            lw.covariance_, index=factor_names, columns=factor_names
        )

        # Track which stocks have valid estimates
        self.fitted_stocks_ = self.loadings_.dropna().index.tolist()

        return self

    def get_covariance(self, stocks: list[str]) -> np.ndarray:
        """Return (N x N) covariance matrix for given stocks.

        Stocks without valid estimates get average idiosyncratic risk.
        """
        # Filter to stocks with valid loadings
        valid = [s for s in stocks if s in self.fitted_stocks_]
        missing = [s for s in stocks if s not in self.fitted_stocks_]

        B = self.loadings_.loc[valid].values.astype(float)
        F = self.factor_cov_.values.astype(float)

        # Factor component
        factor_cov = B @ F @ B.T

        # Idiosyncratic component
        idio = self.idio_var_.loc[valid].values.astype(float)
        idio = np.where(np.isnan(idio), np.nanmedian(idio), idio)
        D = np.diag(idio)

        sigma = factor_cov + D

        # If there are missing stocks, expand matrix with average idiosyncratic
        if missing:
            n_valid = len(valid)
            n_total = len(stocks)
            sigma_full = np.eye(n_total) * np.nanmedian(self.idio_var_)
            # Map valid stocks into full matrix
            valid_idx = [stocks.index(s) for s in valid]
            for i, vi in enumerate(valid_idx):
                for j, vj in enumerate(valid_idx):
                    sigma_full[vi, vj] = sigma[i, j]
            sigma = sigma_full

        # Ensure PSD
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        sigma = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return sigma

    def get_beta(self, stocks: list[str]) -> np.ndarray:
        """Return market beta vector for given stocks."""
        betas = []
        for s in stocks:
            if s in self.fitted_stocks_ and "Mkt-RF" in self.loadings_.columns:
                b = self.loadings_.loc[s, "Mkt-RF"]
                betas.append(b if not np.isnan(b) else 1.0)
            else:
                betas.append(1.0)  # default beta
        return np.array(betas)
