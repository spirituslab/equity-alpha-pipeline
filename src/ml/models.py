"""ML model wrappers for cross-sectional return prediction.

Each model predicts next-month residual return from neutralized signal features.
Training uses pooled cross-sections with purged CV for hyperparameter selection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from src.ml.purged_cv import PurgedKFoldCV


class AlphaModel:
    """Unified interface for cross-sectional return prediction models.

    Usage:
        model = AlphaModel("ridge")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, model_type: str = "ridge", **kwargs):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self._fitted = False

        if model_type == "ridge":
            self.model = RidgeCV(
                alphas=np.logspace(-4, 2, 50),
                **kwargs,
            )
        elif model_type == "elastic_net":
            self.model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9],
                n_alphas=50,
                max_iter=5000,
                **kwargs,
            )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                raise ImportError("XGBoost not installed. Run: uv pip install xgboost")
            self.model = XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 3),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                reg_alpha=kwargs.get("reg_alpha", 0.1),
                reg_lambda=kwargs.get("reg_lambda", 1.0),
                random_state=42,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'ridge', 'elastic_net', or 'xgboost'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AlphaModel":
        """Fit model on training data."""
        # Remove rows with NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]
        y_clean = y[valid]

        if len(X_clean) < 50:
            self._fitted = False
            return self

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        self.model.fit(X_scaled, y_clean)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict alpha scores."""
        if not self._fitted:
            return np.full(len(X), np.nan)

        # Handle NaN in prediction data
        valid = ~np.isnan(X).any(axis=1)
        predictions = np.full(len(X), np.nan)

        if valid.sum() > 0:
            X_scaled = self.scaler.transform(X[valid])
            predictions[valid] = self.model.predict(X_scaled)

        return predictions

    @property
    def feature_importance(self) -> np.ndarray | None:
        """Get feature importance if available."""
        if not self._fitted:
            return None
        if hasattr(self.model, "coef_"):
            return self.model.coef_
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


def ml_combine(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    model_type: str = "ridge",
    train_window: int = 60,
    purge_gap: int = 2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """ML-based signal combination via walk-forward prediction.

    At each rebalance date t:
    1. Collect training data from [t-train_window, t-purge_gap]
    2. Features: neutralized z-scores for all K signals
    3. Target: next-month excess return (over cross-sectional median)
    4. Predict alpha_{i,t+1} for all stocks at time t

    Args:
        signals: dict of signal name -> (date x gvkey) neutralized z-scores
        returns: (date x gvkey) stock returns
        universe: (date x gvkey) boolean mask
        model_type: "ridge", "elastic_net", or "xgboost"
        train_window: months of training data
        purge_gap: months between train and test

    Returns:
        (date x gvkey) alpha predictions
    """
    from tqdm import tqdm

    # Align all signals
    signal_names = sorted(signals.keys())
    common_dates = None
    common_stocks = None
    for sig in signals.values():
        if common_dates is None:
            common_dates = sig.index
            common_stocks = sig.columns
        else:
            common_dates = common_dates.intersection(sig.index)
            common_stocks = common_stocks.intersection(sig.columns)

    # Build feature tensor: (T, N, K)
    T = len(common_dates)
    stocks = common_stocks.tolist()
    N = len(stocks)
    K = len(signal_names)

    features = np.full((T, N, K), np.nan)
    for k, name in enumerate(signal_names):
        features[:, :, k] = signals[name].loc[common_dates, stocks].values

    # Build target: next-month return minus cross-sectional median
    ret_aligned = returns.loc[common_dates, stocks].values
    target = np.full((T, N), np.nan)
    for t in range(T - 1):
        r = ret_aligned[t + 1]
        median_r = np.nanmedian(r)
        target[t] = r - median_r  # predict residual, not raw return

    # Walk-forward prediction
    alpha_predictions = pd.DataFrame(np.nan, index=common_dates, columns=stocks)

    start_idx = train_window + purge_gap
    date_iter = range(start_idx, T - 1)
    if show_progress:
        date_iter = tqdm(date_iter, desc=f"ML combine ({model_type})")

    for t in date_iter:
        # Training window
        train_end = t - purge_gap
        train_start = max(0, train_end - train_window)

        # Pool cross-sections in training window
        X_train_parts = []
        y_train_parts = []
        for tt in range(train_start, train_end):
            # Filter to universe members
            univ_t = universe.loc[common_dates[tt]].reindex(stocks) if common_dates[tt] in universe.index else pd.Series(True, index=stocks)
            mask = univ_t.fillna(False).values.astype(bool)

            X_t = features[tt, mask, :]
            y_t = target[tt, mask]

            # Remove NaN rows
            valid = ~(np.isnan(X_t).any(axis=1) | np.isnan(y_t))
            if valid.sum() > 0:
                X_train_parts.append(X_t[valid])
                y_train_parts.append(y_t[valid])

        if not X_train_parts:
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        if len(X_train) < 100:
            continue

        # Fit model
        model = AlphaModel(model_type)
        model.fit(X_train, y_train)

        if not model._fitted:
            continue

        # Predict for current date
        X_pred = features[t, :, :]
        predictions = model.predict(X_pred)
        alpha_predictions.iloc[t] = predictions

    return alpha_predictions
