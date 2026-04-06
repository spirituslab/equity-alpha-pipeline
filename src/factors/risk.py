import numpy as np
import pandas as pd

from src.factors.base import Factor


class IdiosyncraticVol(Factor):
    """Idiosyncratic Volatility.

    Residual std from 36-month market model regression:
        r_i - rf = alpha + beta * (r_m - rf) + epsilon
        IVOL = std(epsilon)

    Ang et al. (2006): low IVOL stocks outperform high IVOL stocks.
    Signal direction: negative IVOL (lower vol = higher expected return).
    """

    name = "idio_vol"
    category = "risk"

    def __init__(self, window: int = 36):
        self.window = window

    def compute(self, panel) -> pd.DataFrame:
        returns = panel.get_returns()
        market_excess = panel.get_market_excess()
        rf = panel.get_risk_free()

        result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for t_idx in range(self.window, len(returns)):
            t = returns.index[t_idx]
            window_slice = slice(t_idx - self.window, t_idx)
            r_window = returns.iloc[window_slice]
            mkt_window = market_excess.reindex(r_window.index).dropna()
            rf_window = rf.reindex(r_window.index).dropna()

            common_dates = r_window.index.intersection(mkt_window.index).intersection(rf_window.index)
            if len(common_dates) < 24:
                continue

            r_w = r_window.loc[common_dates]
            mkt_w = mkt_window.loc[common_dates].values
            rf_w = rf_window.loc[common_dates].values

            for stock in r_w.columns:
                y = r_w[stock].values - rf_w
                valid = ~np.isnan(y)
                if valid.sum() < 24:
                    continue
                y_valid = y[valid]
                x_valid = mkt_w[valid]
                x_mat = np.column_stack([np.ones(len(x_valid)), x_valid])
                try:
                    beta_hat = np.linalg.lstsq(x_mat, y_valid, rcond=None)[0]
                    residuals = y_valid - x_mat @ beta_hat
                    result.loc[t, stock] = -residuals.std()  # negative: low vol = high signal
                except np.linalg.LinAlgError:
                    continue

        return result
