"""Purged K-Fold Cross-Validation for time-series financial data.

Based on de Prado (2018) "Advances in Financial Machine Learning", Chapter 7.

Standard k-fold CV causes information leakage because temporally adjacent
observations share information (overlapping return windows, autocorrelated
features). Purged CV removes this leakage by:

1. Purging: remove training observations within `purge_gap` periods of test
2. Embargoing: remove a fraction of training observations after each test fold

This is critical for any ML model applied to cross-sectional return prediction.
"""

from typing import Iterator

import numpy as np
import pandas as pd


class PurgedKFoldCV:
    """Purged K-Fold with embargo for time-series financial data.

    Parameters:
        n_splits: number of folds
        purge_gap: number of periods to purge between train and test
        embargo_pct: fraction of training data to embargo after test end
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 2,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        dates: np.ndarray | pd.Index,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) with purging and embargo.

        Args:
            dates: array of period/date labels for each observation.
                   Multiple observations can share the same date
                   (cross-sectional stacking).
        """
        if isinstance(dates, pd.PeriodIndex):
            unique_dates = dates.unique().sort_values()
        else:
            unique_dates = np.unique(dates)
            unique_dates.sort()

        n_dates = len(unique_dates)
        if n_dates < self.n_splits:
            raise ValueError(f"Not enough unique dates ({n_dates}) for {self.n_splits} folds")

        fold_size = n_dates // self.n_splits
        embargo_n = max(1, int(self.embargo_pct * n_dates))

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_dates)

            test_date_set = set(unique_dates[test_start:test_end])

            # Purge: exclude dates within purge_gap of test period boundaries
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_dates, test_end + self.purge_gap)
            purge_date_set = set(unique_dates[purge_start:purge_end])

            # Embargo: exclude dates immediately after test period
            embargo_end = min(n_dates, test_end + embargo_n)
            embargo_date_set = set(unique_dates[test_end:embargo_end])

            excluded = test_date_set | purge_date_set | embargo_date_set

            # Build masks
            if isinstance(dates, pd.PeriodIndex):
                train_mask = ~dates.isin(excluded)
                test_mask = dates.isin(test_date_set)
            else:
                train_mask = np.array([d not in excluded for d in dates])
                test_mask = np.array([d in test_date_set for d in dates])

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class TimeSeriesPurgedSplit:
    """Walk-forward train/test split with purging.

    Unlike PurgedKFoldCV which creates K folds, this creates a single
    expanding or rolling window split suitable for walk-forward validation.

    At each step:
    - Train: [start, t - purge_gap]
    - Test: [t, t + test_size]
    """

    def __init__(
        self,
        train_size: int = 60,
        test_size: int = 12,
        purge_gap: int = 2,
        step: int = 12,
        expanding: bool = False,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.step = step
        self.expanding = expanding

    def split(
        self,
        dates: np.ndarray | pd.Index,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for walk-forward splits."""
        unique_dates = np.unique(dates) if not isinstance(dates, pd.PeriodIndex) else dates.unique().sort_values()
        n_dates = len(unique_dates)

        start = self.train_size
        while start + self.test_size <= n_dates:
            test_start = start
            test_end = min(start + self.test_size, n_dates)
            test_date_set = set(unique_dates[test_start:test_end])

            if self.expanding:
                train_end = test_start - self.purge_gap
                train_start = 0
            else:
                train_end = test_start - self.purge_gap
                train_start = max(0, train_end - self.train_size)

            train_date_set = set(unique_dates[train_start:train_end])

            if isinstance(dates, pd.PeriodIndex):
                train_mask = dates.isin(train_date_set)
                test_mask = dates.isin(test_date_set)
            else:
                train_mask = np.array([d in train_date_set for d in dates])
                test_mask = np.array([d in test_date_set for d in dates])

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

            start += self.step
