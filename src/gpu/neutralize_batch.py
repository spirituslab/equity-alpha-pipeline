"""GPU-accelerated neutralization with projection matrix caching.

Key insight: the control variables (sector, size, beta) are the SAME
for all signals at each date. So we precompute the projection matrix
M_t = I - X(X'X)^{-1}X' once, then neutralizing any signal is just
a matrix-vector multiply: residual = M_t @ y.

This makes stepwise selection fast: adding a new signal to the trial
set only requires one mat-vec multiply per date, not a full OLS.
"""

import numpy as np
import pandas as pd

from src.gpu.backend import GPU_AVAILABLE, to_gpu, to_cpu

if GPU_AVAILABLE:
    import cupy as cp


class ProjectionCache:
    """Precomputed projection matrices for fast neutralization.

    M_t = I - X_t @ (X_t' X_t)^{-1} @ X_t'

    After construction, neutralizing any signal is:
        residual_t = M_t @ signal_t
    """

    def __init__(self):
        self.projections = None  # (T, N, N) or list of (N_t, N_t)
        self.dates = None
        self.stock_masks = None  # which stocks are valid at each date
        self.stocks = None
        self._built = False

    def build(
        self,
        sector_labels: pd.DataFrame,
        log_mcap: pd.DataFrame,
        beta: pd.DataFrame,
        neutralize_sector: bool = True,
        neutralize_size: bool = True,
        neutralize_beta: bool = True,
        min_obs: int = 50,
    ):
        """Precompute projection matrices for all dates."""
        # Find common dates
        dates = log_mcap.index
        stocks = log_mcap.columns
        self.dates = dates
        self.stocks = stocks
        T = len(dates)
        N = len(stocks)

        self.projections = {}
        self.stock_masks = {}

        for t_idx, t in enumerate(dates):
            # Build control matrix X for this date
            x_parts = []

            if neutralize_size and t in log_mcap.index:
                lmc = log_mcap.loc[t].values
                valid_size = ~np.isnan(lmc)
            else:
                valid_size = np.ones(N, dtype=bool)

            if neutralize_beta and t in beta.index:
                b = beta.loc[t].reindex(stocks).values if hasattr(beta.loc[t], 'reindex') else beta.loc[t].values
                valid_beta = ~np.isnan(b)
            else:
                b = None
                valid_beta = np.ones(N, dtype=bool)

            # Combined valid mask
            valid = valid_size & valid_beta

            if neutralize_sector and t in sector_labels.index:
                sec = sector_labels.loc[t].reindex(stocks) if hasattr(sector_labels.loc[t], 'reindex') else sector_labels.loc[t]
                sec_valid = sec.notna()
                valid = valid & sec_valid.values

            valid_idx = np.where(valid)[0]
            if len(valid_idx) < min_obs:
                continue

            # Build X matrix for valid stocks
            x_cols = [np.ones(len(valid_idx))]  # intercept

            if neutralize_sector and t in sector_labels.index:
                sec_vals = sec.iloc[valid_idx] if hasattr(sec, 'iloc') else sec[valid_idx]
                dummies = pd.get_dummies(sec_vals, drop_first=True, dtype=float)
                if len(dummies.columns) > 0:
                    x_cols.append(dummies.values)

            if neutralize_size:
                x_cols.append(lmc[valid_idx].reshape(-1, 1))

            if neutralize_beta and b is not None:
                x_cols.append(b[valid_idx].reshape(-1, 1))

            X = np.column_stack(x_cols).astype(np.float64)
            n_valid = len(valid_idx)

            # Compute projection matrix: M = I - X(X'X)^{-1}X'
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                hat = X @ XtX_inv @ X.T
                M = np.eye(n_valid) - hat
            except np.linalg.LinAlgError:
                continue

            self.projections[t] = M.astype(np.float32)
            self.stock_masks[t] = valid_idx

        self._built = True
        return self

    def neutralize_fast(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Neutralize a signal using precomputed projection matrices.

        Just a matrix-vector multiply per date — no OLS needed.
        """
        if not self._built:
            raise RuntimeError("ProjectionCache not built. Call build() first.")

        result = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns, dtype=float)

        for t in signal.index:
            if t not in self.projections:
                continue

            M = self.projections[t]
            valid_idx = self.stock_masks[t]
            stocks = self.stocks

            y = signal.loc[t].reindex(stocks).values
            y_valid = y[valid_idx]

            # Skip if too many NaN in signal
            nan_mask = np.isnan(y_valid)
            if nan_mask.sum() > len(y_valid) * 0.5:
                continue

            # Replace NaN with 0 for projection (they'll be NaN in output)
            y_clean = np.where(nan_mask, 0, y_valid).astype(np.float32)

            # Project
            residual = M @ y_clean

            # Restore NaN
            residual[nan_mask] = np.nan

            result.loc[t, stocks[valid_idx]] = residual

        return result

    def neutralize_fast_gpu(self, signal: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated neutralization using cached projections."""
        if not GPU_AVAILABLE:
            return self.neutralize_fast(signal)

        if not self._built:
            raise RuntimeError("ProjectionCache not built. Call build() first.")

        result = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns, dtype=float)

        for t in signal.index:
            if t not in self.projections:
                continue

            M_gpu = cp.asarray(self.projections[t])
            valid_idx = self.stock_masks[t]
            stocks = self.stocks

            y = signal.loc[t].reindex(stocks).values
            y_valid = y[valid_idx]

            nan_mask = np.isnan(y_valid)
            if nan_mask.sum() > len(y_valid) * 0.5:
                continue

            y_clean = cp.asarray(np.where(nan_mask, 0, y_valid).astype(np.float32))
            residual = to_cpu(M_gpu @ y_clean)
            residual[nan_mask] = np.nan

            result.loc[t, stocks[valid_idx]] = residual

        return result
