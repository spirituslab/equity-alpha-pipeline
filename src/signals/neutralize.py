"""Cross-sectional signal neutralization.

Removes mechanical correlation between signals and sector/size/beta,
ensuring the final alpha score captures pure stock selection (alpha)
rather than factor tilts.

For each signal s and each date t, run cross-sectional OLS:
    s_{i,t} = a + b1*sector_dummies + b2*log(mcap_{i,t}) + b3*beta_{i,t} + epsilon_{i,t}

The residual epsilon is the neutralized signal.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def neutralize_signal(
    signal: pd.DataFrame,
    sector_labels: pd.DataFrame,
    log_mcap: pd.DataFrame,
    beta: pd.DataFrame,
    neutralize_sector: bool = True,
    neutralize_size: bool = True,
    neutralize_beta: bool = True,
    min_obs: int = 50,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Cross-sectional neutralization via OLS regression residuals.

    For each date t, regresses the signal on control variables and
    returns the residual as the neutralized signal.

    Args:
        signal: (date x gvkey) z-scored signal values
        sector_labels: (date x gvkey) sector label strings
        log_mcap: (date x gvkey) log market cap
        beta: (date x gvkey) market beta
        neutralize_sector: include sector dummies as controls
        neutralize_size: include log(mcap) as control
        neutralize_beta: include beta as control
        min_obs: minimum stocks required for regression

    Returns:
        (date x gvkey) neutralized signal (OLS residuals)
    """
    neutralized = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)

    dates = tqdm(signal.index, desc="Neutralizing", disable=not show_progress)

    for t in dates:
        y = signal.loc[t].dropna()
        if len(y) < min_obs:
            continue

        stocks = y.index

        # Build control matrix
        x_parts = []

        if neutralize_sector:
            sec = sector_labels.loc[t].reindex(stocks) if t in sector_labels.index else None
            if sec is not None:
                sec = sec.dropna()
                # One-hot encode, drop first for identification
                dummies = pd.get_dummies(sec, drop_first=True, dtype=float)
                if len(dummies.columns) > 0:
                    x_parts.append(dummies)

        if neutralize_size:
            lmc = log_mcap.loc[t].reindex(stocks) if t in log_mcap.index else None
            if lmc is not None:
                lmc = lmc.dropna()
                x_parts.append(lmc.rename("log_mcap").to_frame())

        if neutralize_beta:
            b = beta.loc[t].reindex(stocks) if t in beta.index else None
            if b is not None:
                b = b.dropna()
                x_parts.append(b.rename("beta").to_frame())

        if not x_parts:
            # No controls — signal is already "neutralized"
            neutralized.loc[t, stocks] = y
            continue

        # Align all inputs on common stocks
        X = pd.concat(x_parts, axis=1)
        common = y.index.intersection(X.dropna().index)
        if len(common) < min_obs:
            continue

        y_common = y.loc[common].values
        X_common = X.loc[common].values

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(common)), X_common])

        # OLS: y = X @ beta + epsilon, return epsilon
        try:
            beta_hat, _, _, _ = np.linalg.lstsq(X_with_const, y_common, rcond=None)
            residuals = y_common - X_with_const @ beta_hat
            neutralized.loc[t, common] = residuals
        except np.linalg.LinAlgError:
            neutralized.loc[t, common] = y.loc[common]

    return neutralized


def neutralize_all_signals(
    signals: dict[str, pd.DataFrame],
    sector_labels: pd.DataFrame,
    log_mcap: pd.DataFrame,
    beta: pd.DataFrame,
    neutralize_sector: bool = True,
    neutralize_size: bool = True,
    neutralize_beta: bool = True,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Neutralize all signals against sector/size/beta controls.

    Args:
        signals: dict mapping signal name -> (date x gvkey) z-scored DataFrame
        Other args: same as neutralize_signal

    Returns:
        dict mapping signal name -> (date x gvkey) neutralized DataFrame
    """
    neutralized = {}
    for name, sig in signals.items():
        print(f"  Neutralizing {name}...")
        neutralized[name] = neutralize_signal(
            sig,
            sector_labels=sector_labels,
            log_mcap=log_mcap,
            beta=beta,
            neutralize_sector=neutralize_sector,
            neutralize_size=neutralize_size,
            neutralize_beta=neutralize_beta,
            show_progress=show_progress,
        )
    return neutralized


def verify_neutralization(
    neutralized_signal: pd.DataFrame,
    sector_labels: pd.DataFrame,
    log_mcap: pd.DataFrame,
    beta: pd.DataFrame,
    sample_dates: int = 20,
) -> dict[str, float]:
    """Verify that neutralized signal has near-zero correlation with controls.

    Samples random dates and computes average absolute correlation
    with sector dummies, log(mcap), and beta.

    Returns dict with average absolute correlations (should all be < 0.05).
    """
    dates = neutralized_signal.dropna(how="all").index
    if len(dates) > sample_dates:
        rng = np.random.default_rng(42)
        dates = rng.choice(dates, size=sample_dates, replace=False)

    corr_size = []
    corr_beta = []
    corr_sector = []

    for t in dates:
        sig = neutralized_signal.loc[t].dropna()
        if len(sig) < 20:
            continue

        # Size correlation
        if t in log_mcap.index:
            lmc = log_mcap.loc[t].reindex(sig.index).dropna()
            common = sig.index.intersection(lmc.index)
            if len(common) > 20:
                corr_size.append(abs(np.corrcoef(sig.loc[common], lmc.loc[common])[0, 1]))

        # Beta correlation
        if t in beta.index:
            b = beta.loc[t].reindex(sig.index).dropna()
            common = sig.index.intersection(b.index)
            if len(common) > 20:
                corr_beta.append(abs(np.corrcoef(sig.loc[common], b.loc[common])[0, 1]))

        # Sector: average absolute correlation with each sector dummy
        if t in sector_labels.index:
            sec = sector_labels.loc[t].reindex(sig.index).dropna()
            common = sig.index.intersection(sec.index)
            if len(common) > 20:
                for sector in sec.unique():
                    dummy = (sec.loc[common] == sector).astype(float)
                    if dummy.std() > 0:
                        corr_sector.append(abs(np.corrcoef(sig.loc[common], dummy)[0, 1]))

    return {
        "avg_abs_corr_size": np.mean(corr_size) if corr_size else np.nan,
        "avg_abs_corr_beta": np.mean(corr_beta) if corr_beta else np.nan,
        "avg_abs_corr_sector": np.mean(corr_sector) if corr_sector else np.nan,
    }
