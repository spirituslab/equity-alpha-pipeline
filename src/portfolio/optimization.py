"""Dollar-neutral, beta-neutral, sector-neutral portfolio optimizer.

Solves:
    max   alpha' @ w - (lambda/2) * w' @ Sigma @ w - kappa * ||w - w_prev||_1
    s.t.  sum(w) = 0                              (dollar-neutral)
          |beta' @ w| <= max_beta                  (beta-neutral)
          |sum(w[sector==s])| <= max_sector_net    (sector-neutral)
          ||w||_1 <= max_gross                     (leverage cap)
          -max_wt <= w_i <= max_wt                 (position limits)
"""

import numpy as np
import cvxpy as cp


def long_short_optimize(
    alpha: np.ndarray,
    sigma: np.ndarray,
    beta: np.ndarray,
    sector: np.ndarray,
    w_prev: np.ndarray = None,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.005,
    max_stock_weight: float = 0.02,
    max_sector_net: float = 0.05,
    max_gross_leverage: float = 2.0,
    max_beta_exposure: float = 0.05,
) -> np.ndarray:
    """Constrained long/short portfolio optimization.

    Args:
        alpha: (N,) alpha scores
        sigma: (N, N) covariance matrix
        beta: (N,) market betas
        sector: (N,) integer sector labels (0..S-1)
        w_prev: (N,) previous weights (for turnover penalty)
        risk_aversion: lambda in objective
        turnover_penalty: kappa for ||w - w_prev||_1
        max_stock_weight: per-position limit (both long and short)
        max_sector_net: max absolute net sector exposure
        max_gross_leverage: max sum of absolute weights
        max_beta_exposure: max absolute portfolio beta

    Returns:
        (N,) optimal weight vector
    """
    N = len(alpha)

    if w_prev is None:
        w_prev = np.zeros(N)

    w = cp.Variable(N)

    # Objective
    ret = alpha @ w
    risk = cp.quad_form(w, sigma)
    turnover = cp.norm(w - w_prev, 1)
    objective = cp.Maximize(ret - (risk_aversion / 2) * risk - turnover_penalty * turnover)

    # Constraints
    constraints = [
        cp.sum(w) == 0,                          # Dollar-neutral
        cp.abs(beta @ w) <= max_beta_exposure,   # Beta-neutral
        cp.norm(w, 1) <= max_gross_leverage,     # Leverage cap
        w >= -max_stock_weight,                  # Position limits
        w <= max_stock_weight,
    ]

    # Sector neutrality
    unique_sectors = np.unique(sector[~np.isnan(sector)])
    for s in unique_sectors:
        mask = (sector == s).astype(float)
        constraints.append(cp.abs(mask @ w) <= max_sector_net)

    prob = cp.Problem(objective, constraints)

    # Try SCS first (handles SOCP well)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    except cp.SolverError:
        pass

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        # Fallback: relax constraints
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-5)
        except cp.SolverError:
            pass

    if prob.status in ["infeasible", "unbounded"] or w.value is None:
        # Last resort: return equal-weight naive portfolio
        return _naive_long_short(alpha, max_stock_weight)

    return w.value


def _naive_long_short(alpha: np.ndarray, max_weight: float = 0.02) -> np.ndarray:
    """Fallback: simple long/short based on alpha ranks."""
    N = len(alpha)
    n_positions = min(50, N // 4)
    if n_positions < 2:
        return np.zeros(N)

    ranked = np.argsort(alpha)
    weights = np.zeros(N)
    weights[ranked[-n_positions:]] = 1.0 / n_positions
    weights[ranked[:n_positions]] = -1.0 / n_positions
    return weights
