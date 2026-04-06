"""Signal-to-portfolio mapping.

Converts composite alpha scores into target portfolio weights.
"""

import numpy as np
import pandas as pd


def signal_to_target_weights(
    alpha_scores: pd.Series,
    n_long: int = 50,
    n_short: int = 50,
    weight_method: str = "alpha_proportional",
) -> pd.Series:
    """Convert alpha scores to target portfolio weights.

    1. Rank stocks by alpha score
    2. Long top n_long, short bottom n_short
    3. Assign weights within each leg

    Args:
        alpha_scores: alpha_i for each stock (higher = more attractive)
        n_long: number of long positions
        n_short: number of short positions
        weight_method: "equal_weight" or "alpha_proportional"

    Returns:
        pd.Series of weights, dollar-neutral (sum ≈ 0)
    """
    scores = alpha_scores.dropna().sort_values()
    n = len(scores)

    if n < n_long + n_short:
        # Not enough stocks — use top/bottom halves
        n_short = n // 4
        n_long = n // 4

    if n_long == 0 or n_short == 0:
        return pd.Series(0.0, index=alpha_scores.index)

    short_stocks = scores.index[:n_short]
    long_stocks = scores.index[-n_long:]

    weights = pd.Series(0.0, index=alpha_scores.index)

    if weight_method == "equal_weight":
        weights.loc[long_stocks] = 1.0 / n_long
        weights.loc[short_stocks] = -1.0 / n_short

    elif weight_method == "alpha_proportional":
        # Long leg: weight proportional to alpha magnitude
        long_alpha = scores.loc[long_stocks]
        long_alpha = long_alpha - long_alpha.min() + 1e-8  # shift to positive
        weights.loc[long_stocks] = long_alpha / long_alpha.sum()

        # Short leg: weight proportional to (negative) alpha magnitude
        short_alpha = -scores.loc[short_stocks]
        short_alpha = short_alpha - short_alpha.min() + 1e-8
        weights.loc[short_stocks] = -short_alpha / short_alpha.sum()
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    return weights
