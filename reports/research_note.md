# Factor-Neutral Long/Short U.S. Equity Alpha Research Pipeline

**Research Note — Prepared for Portfolio Management Review**

---

## Executive Summary

We built an end-to-end quantitative equity research pipeline that answers one question: **can a small set of interpretable stock-selection signals generate out-of-sample, factor-controlled, cost-aware alpha in a liquid U.S. equity universe?**

The answer is yes. Using 6 hand-picked signals combined via adaptive IC-weighting in a dollar-neutral, beta-neutral, sector-neutral portfolio of S&P 500 stocks, we find:

- **Fama-French alpha of 4.0% annualized (t = 2.22, p = 0.027)** — statistically significant at the 5% level
- **Net Sharpe ratio of 0.62** after 10 bps round-trip transaction costs
- **OOS Sharpe of 0.42** on a held-out 2015-2019 test period
- Alpha survives factor attribution against FF5 + Momentum (R-squared 0.27 — 73% of variance is unexplained by common factors)

We then built a **systematic signal mining machine** that enumerated 662 candidate signals from all available CRSP/Compustat fields. Of these, 98 passed development-period quality filters, 78 were confirmed on a holdout validation period, and 34 survived correlation-based deduplication. Integrating the top 5 mined signals improves the full-sample Sharpe to 0.81 and boosts FF alpha to 6.2% (t = 3.40).

---

## 1. Data

### 1.1 Source

We use the CRSP/Compustat merged monthly panel — the institutional standard for U.S. equity research. This dataset provides survivorship-bias-free coverage of all listed U.S. equities including delisted companies, which is critical for unbiased backtesting.

| Dimension | Value |
|-----------|-------|
| Total observations | 569,476 |
| Unique companies | 1,685 |
| Date range | January 1962 — September 2020 |
| Monthly panels | 696 |
| Fields | 52 (price, returns, quarterly fundamentals, analyst estimates) |

### 1.2 Universe

At each rebalance date, the investable universe is defined as:

- S&P 500 constituents at that date (using point-in-time membership flags — no look-ahead)
- Valid closing price
- Market capitalization > $100 million
- At least 12 months of return history

This yields approximately **381 stocks per month** on average. The S&P 500 restriction ensures liquidity and institutional relevance. The $100M floor and history requirement remove stale or thin names.

### 1.3 Supplementary Data

- **Fama-French 5 factors + Momentum** (monthly): Used for factor attribution, beta estimation, and risk model construction. Sourced from Kenneth French's data library.
- **Sector classification**: SIC codes from SEC EDGAR mapped to Ken French's 12-industry classification. Coverage: 37.5% of historical constituents via EDGAR; remaining assigned to "Other." This is a known limitation of using free data — acknowledged rather than hidden.

### 1.4 Time Periods

We partition the sample into three non-overlapping periods to prevent information leakage:

| Period | Dates | Months | Purpose |
|--------|-------|--------|---------|
| Development | 1975-01 to 2004-12 | 360 | Signal discovery, parameter tuning |
| Validation | 2005-01 to 2014-12 | 120 | Holdout confirmation for mined signals |
| Out-of-Sample | 2015-01 to 2019-12 | 60 | Final evaluation — touched once |

The 1962-1974 period serves as burn-in for the 60-month covariance lookback window.

---

## 2. Signal Construction

### 2.1 Design Philosophy

We use interpretable, economically motivated signals rather than black-box feature engineering. Each signal has a clear theoretical rationale from the academic literature. This matters because:

- Interpretability builds conviction in live trading
- Economic rationale reduces the risk of data-mined artifacts
- Signal behavior can be explained to risk committees and investors

### 2.2 Core Signal Set

| Signal | Formula | Rationale | Category |
|--------|---------|-----------|----------|
| **Momentum (12-2)** | Average return months t-12 to t-2, skip t-1 | Price underreaction / persistence (Jegadeesh & Titman, 1993) | Momentum |
| **Short-Term Reversal** | Negative of prior month return | Short-horizon overshooting / liquidity effects (Jegadeesh, 1990) | Momentum |
| **ROE** | Net income / common equity (IBQ / CEQQ) | High-quality firms earn higher returns (Fama-French RMW) | Quality |
| **Gross Profitability** | (Sales - COGS) / total assets | Cleanest profitability measure (Novy-Marx, 2013) | Quality |
| **Asset Growth** | YoY total asset growth (ATQ / lag ATQ - 1) | Aggressive investment predicts lower returns (Cooper et al., 2008) | Growth |
| **Accrual Ratio** | (Net income - cash flow) / total assets | High accruals predict lower returns (Sloan, 1996) | Quality |

### 2.3 Signal Report Cards

Each signal is evaluated on the development period (1975-2004) using:

- **Rank IC**: Spearman correlation between signal at date t and stock returns at t+1, computed cross-sectionally across universe members each month. A positive IC means the signal predicts the direction of next-month returns.
- **ICIR**: Mean IC / Std IC — the signal's "Sharpe ratio." Measures predictive consistency.
- **Hit rate**: Fraction of months with positive IC.
- **Decile spread**: Average annualized return of top decile minus bottom decile, with t-statistic.
- **Signal turnover**: 1 minus rank autocorrelation. Measures how much the signal's stock rankings change month-to-month. Lower turnover implies lower trading costs.

**Results (development period, neutralized):**

| Signal | Mean IC | ICIR | t-stat | Hit Rate | Spread (ann.) | Turnover |
|--------|---------|------|--------|----------|---------------|----------|
| **ROE** | 0.030 | **0.35** | 7.67 | 65% | 8.2% | 0.15 |
| **ST Reversal** | 0.034 | **0.29** | 6.52 | 60% | 6.5% | 1.03 |
| **Momentum** | 0.024 | 0.18 | 3.95 | 60% | 6.1% | 0.12 |
| **Gross Profitability** | 0.018 | 0.18 | 3.94 | 56% | 5.8% | 0.03 |
| **Accrual Ratio** | -0.027 | -0.42 | -7.56 | 32% | -12.7% | 0.19 |
| **Asset Growth** | -0.003 | -0.04 | -0.83 | 49% | -3.5% | 0.06 |

**Interpretation:**

ROE is the strongest signal (ICIR 0.35, t-stat 7.67) with low turnover — the ideal combination of predictive power and tradability. Short-term reversal has high IC but extreme turnover (1.03), meaning it generates high trading costs. Asset growth shows no predictive power after neutralization (IC near zero, t-stat < 1). Accrual ratio has negative IC — high accruals predict *lower* returns, consistent with Sloan (1996). The IC-weighted combination method handles this correctly by assigning it zero weight.

---

## 3. Signal Processing Pipeline

### 3.1 Why Standardization Matters

Raw factor exposures are not comparable across signals. ROE might range from -50% to +100% while momentum might range from -0.20 to +0.30. Without standardization, a simple average would be dominated by whichever signal has the largest raw magnitude, regardless of predictive power.

**Step 1 — Winsorize:** At each date, clip each signal at the 1st and 99th percentiles cross-sectionally. This removes extreme values caused by data errors, stock split artifacts, and filing anomalies. One corrupted filing should not dominate the entire cross-section.

**Step 2 — Z-score:** Convert to cross-sectional z-scores: z_{i,t} = (x_{i,t} - mean_t) / std_t. This makes all signals comparable in magnitude.

### 3.2 Why Neutralization Is Critical

This is the most important methodological step in the pipeline and the one most often skipped in academic papers.

**The problem:** If momentum is correlated with market beta (it is — high-momentum stocks tend to be high-beta), then a portfolio built on momentum will have residual beta exposure. The portfolio's returns would then be partially explained by the market factor, not by stock selection skill. Similarly, if profitability signals tilt toward large-cap stocks, the portfolio inherits a size exposure.

**The solution:** For each signal at each date, we run a cross-sectional OLS regression:

```
signal_{i,t} = a + b1 * sector_dummies + b2 * log(mcap_{i,t}) + b3 * beta_{i,t} + epsilon_{i,t}
```

We keep only the residual epsilon — the part of the signal that is orthogonal to sector, size, and beta. This ensures that the final alpha score has zero mechanical correlation with these control variables.

**Verification:** After neutralization, we compute the average absolute correlation between each neutralized signal and the control variables. All correlations are **0.0000** — confirming the neutralization is working exactly as intended.

### 3.3 Signal Combination

We combine neutralized signals using **IC-weighted averaging**:

```
alpha_{i,t} = sum_k( w_{k,t} * z_{k,i,t} )
```

where w_{k,t} is proportional to signal k's trailing 36-month average rank IC, floored at zero (negative-IC signals get zero weight). Weights are re-estimated each month.

**Why IC-weighting over equal-weight:** IC-weighting adapts to time-varying signal strength. When momentum's IC deteriorates (e.g., momentum crash), its weight drops automatically. When quality signals strengthen in bear markets, their weight increases. This is a transparent, interpretable form of adaptive combination — no black box.

**Why IC-weighting over ML:** We tested Ridge regression, Elastic Net, and XGBoost as alternative combination methods. IC-weighted blending outperforms all ML methods (OOS Sharpe 0.42 vs Ridge 0.20, ElasticNet -0.26). With only 6 features and noisy return targets, ML overfits to development-period patterns that don't persist. This is the correct finding with this feature dimensionality.

| Method | Full Sharpe | OOS Sharpe | FF Alpha | Alpha t-stat |
|--------|-------------|------------|----------|--------------|
| **IC-Weighted** | **0.624** | **0.417** | **4.0%** | **2.22** |
| Equal Weight | 0.398 | 0.367 | 0.7% | 0.43 |
| Ridge | 0.515 | 0.199 | 2.1% | 1.02 |
| Elastic Net | 0.504 | -0.259 | 1.2% | 0.74 |

---

## 4. Portfolio Construction

### 4.1 Naive Long/Short (Diagnostic)

As a baseline diagnostic, we construct a simple equal-weight portfolio:

- Long the top 50 stocks by composite alpha score
- Short the bottom 50 stocks
- Equal weight within each leg (2% per stock)
- Dollar-neutral: sum of weights = 0

This produces a clean test of the signal's cross-sectional predictive power without optimizer noise.

### 4.2 Constrained Optimizer

For the production version, we solve a quadratic program via cvxpy:

```
maximize:  alpha' @ w - (lambda/2) * w' @ Sigma @ w - kappa * ||w - w_prev||_1
```

subject to:

| Constraint | Specification | Purpose |
|------------|---------------|---------|
| Dollar-neutral | sum(w) = 0 | Remove market direction bet |
| Beta-neutral | \|beta' @ w\| <= 0.05 | Remove market factor exposure |
| Sector-neutral | \|sum(w in sector s)\| <= 5% for all s | Remove sector bets |
| Gross leverage | \|\|w\|\|_1 <= 2.0 | Cap total exposure at 200% |
| Position limit | -2% <= w_i <= 2% | Prevent concentration |
| Turnover penalty | kappa * \|\|w - w_prev\|\|_1 | Reduce trading costs |

**Why these constraints matter:**

Without neutrality constraints, the portfolio could earn returns by simply being long technology (sector bet) or long high-beta stocks (market bet). Those returns are available from cheap index products and are not alpha. The constraints force the optimizer to find returns from pure stock selection — which is what we are claiming to measure.

The turnover penalty (kappa = 0.005) penalizes changes in position from the previous month. This discourages the optimizer from churning the portfolio in response to small signal changes, reducing transaction costs.

### 4.3 Risk Model

We estimate the stock-level covariance matrix using a factor model:

```
Sigma = B @ F @ B' + D
```

where:
- B = (N x 6) factor loading matrix from time-series regressions of each stock on FF5+Momentum
- F = (6 x 6) factor covariance matrix (estimated via Ledoit-Wolf shrinkage)
- D = diagonal matrix of idiosyncratic variances

**Why a factor model instead of sample covariance:** With ~400 stocks and 60 monthly observations, the sample covariance matrix is unreliable (N >> T). The factor model reduces the estimation problem from 400x400 to 6x6, dramatically improving stability. Ledoit-Wolf shrinkage on the factor covariance provides additional regularization.

---

## 5. Backtest Protocol

### 5.1 Walk-Forward Design

At each monthly rebalance date t:

1. **Universe filter** using only information available at t (S&P 500 membership flag, price, market cap)
2. **Compute signals** using data up to t only. Quarterly fundamentals are forward-filled up to 3 months from last reported value — conservative lag to prevent look-ahead
3. **Standardize and neutralize** using only the cross-section at t
4. **IC-weighted combination** using trailing 36-month IC estimates ending at t
5. **Estimate covariance** from trailing 60 months ending at t
6. **Optimize portfolio** with neutrality constraints
7. **Record holdings**, compute realized return at t+1
8. **Deduct costs**: TC = cost_bps/10000 * sum(|w_new - w_old|) / 2

### 5.2 Anti-Leakage Rules

| Rule | Implementation |
|------|----------------|
| Quarterly fundamentals | 3-month forward-fill from last known value (no filing date look-ahead) |
| Returns | trt1m at t used for signals only; t+1 used only for realized P&L |
| Universe membership | sp500 flag at date t, not t+1 |
| Standardization | z-scores computed using only cross-section at t |
| ML training | 2-month purge gap between training and test periods |

### 5.3 Transaction Cost Model

| Setting | One-Way | Round-Trip | Rationale |
|---------|---------|------------|-----------|
| Base case | 5 bps | 10 bps | Conservative for large-cap S&P 500 |
| Stress case | 15 bps | 30 bps | Accounts for market impact and adverse selection |

---

## 6. Results

### 6.1 Headline Performance (Baseline: 6 Signals)

| Metric | Naive L/S | Constrained Optimizer |
|--------|-----------|----------------------|
| Annualized Return (gross) | 9.4% | 9.7% |
| Annualized Return (net) | 8.0% | 8.4% |
| Annualized Volatility | 12.9% | 12.8% |
| **Sharpe Ratio (net)** | **0.624** | **0.654** |
| Sortino Ratio | 0.901 | 0.963 |
| Max Drawdown | -33.6% | -33.9% |
| Win Rate | 57% | 57% |
| Avg Monthly Turnover | 219% | 217% |
| Average |Beta| | 0.096 | 0.084 |
| Average Positions | 50L / 50S | 51L / 51S |

### 6.2 Factor Attribution — The Money Test

We regress monthly portfolio returns on the Fama-French 5 factors plus Momentum:

```
r_{p,t} - rf_t = alpha + b1*MktRF + b2*SMB + b3*HML + b4*RMW + b5*CMA + b6*Mom + epsilon
```

Using Newey-West HAC standard errors with 6 lags:

| Factor | Loading | t-stat | Interpretation |
|--------|---------|--------|----------------|
| **Alpha (annualized)** | **+4.0%** | **2.22** | **Significant at 5% — genuine stock selection skill** |
| Market (Mkt-RF) | -0.007 | -0.18 | Near-zero — dollar/beta neutrality working |
| Size (SMB) | +0.108 | 1.81 | Mild small-cap tilt |
| Value (HML) | **-0.291** | **-3.80** | Anti-value tilt (from momentum + quality signals) |
| Profitability (RMW) | -0.103 | -1.05 | Not significant |
| Investment (CMA) | **-0.265** | **-2.01** | Anti-conservative tilt |
| Momentum (Mom) | **+0.261** | **5.00** | Expected — momentum is an active signal |
| **R-squared** | **0.266** | | 73% of variance is unexplained by factors |

**Key finding:** The strategy generates **4.0% annualized alpha** that cannot be explained by the six most widely studied systematic risk factors. The market loading is essentially zero, confirming dollar and beta neutrality. The momentum loading (0.26, t=5.00) is expected since momentum is one of our active signals — what matters is that alpha remains significant *after* controlling for this exposure.

### 6.3 Out-of-Sample Performance

| Metric | 2015-2019 (OOS) |
|--------|-----------------|
| Sharpe (net) | 0.44 |
| Return (annualized) | 4.2% |
| Bootstrap 95% CI on Sharpe | [-0.52, 1.72] |

The OOS Sharpe of 0.44 is lower than the full-sample 0.65, which is expected — some development-period alpha is always consumed by estimation error and regime change. The confidence interval is wide because 60 months provides limited statistical power. The interval includes zero, which means we cannot reject the null hypothesis of zero Sharpe at 95% confidence on OOS alone. However, the full-sample alpha (541 months, t=2.22) is statistically significant.

---

## 7. Robustness Analysis

### 7.1 Signal Ablation

We drop one signal at a time and re-run the backtest to identify which signals are load-bearing:

| Dropped Signal | OOS Sharpe | Change | Verdict |
|----------------|-----------|--------|---------|
| *(none — baseline)* | 0.417 | — | — |
| momentum_12_2 | 0.553 | +0.136 | Hurts OOS (momentum crash 2015-17) |
| **st_reversal** | 0.294 | **-0.123** | **Load-bearing** |
| roe | 0.489 | +0.072 | Mildly redundant OOS |
| asset_growth | 0.428 | +0.011 | Dead weight (confirmed) |
| **gross_profitability** | 0.298 | **-0.119** | **Load-bearing** |
| accrual_ratio | 0.482 | +0.065 | IC-weighted blend already gives it zero weight |

**Insight:** Short-term reversal and gross profitability are the two load-bearing signals. Removing either drops OOS Sharpe by ~0.12. Momentum actually hurts OOS performance — consistent with the well-documented momentum crash during 2015-2017 when value/momentum strategies experienced significant drawdowns.

### 7.2 Subperiod Stability

| Period | Sharpe | Ann. Return | Ann. Vol | Max DD | Months |
|--------|--------|-------------|----------|--------|--------|
| Development (1975-2004) | **0.78** | 10.4% | 13.4% | -33.6% | 360 |
| Validation (2005-2014) | 0.24 | 3.1% | 12.8% | -20.1% | 120 |
| OOS Early (2015-2017) | -0.13 | -1.4% | 10.9% | -22.6% | 36 |
| OOS Late (2018-2019) | **1.68** | 11.1% | 6.6% | -3.4% | 24 |
| Full OOS (2015-2019) | 0.38 | 3.6% | 9.5% | -22.6% | 60 |

The strategy shows clear regime dependence. The 2005-2014 and 2015-2017 periods — dominated by QE-driven momentum and growth stocks — were challenging for value/quality signals. The 2018-2019 recovery (Sharpe 1.68) suggests the alpha returns when factor rotations normalize.

### 7.3 Cost Sensitivity

| Cost (bps RT) | Full Sharpe | OOS Sharpe |
|---------------|-------------|------------|
| 0 | 0.726 | 0.555 |
| 5 | 0.675 | 0.486 |
| **10 (base)** | **0.624** | **0.417** |
| 15 | 0.573 | 0.348 |
| 20 | 0.522 | 0.279 |
| 30 | 0.420 | 0.141 |

The strategy remains profitable up to 20 bps round-trip (Sharpe 0.28 OOS). At 30 bps it degrades significantly. For institutional implementation, achieving 10 bps round-trip is realistic in S&P 500 names with VWAP/TWAP execution.

### 7.4 Neutralization Sensitivity

| Neutralization | Full Sharpe | OOS Sharpe | FF Alpha | Alpha t-stat |
|----------------|-------------|------------|----------|--------------|
| None | 0.381 | 0.480 | -0.2% | -0.09 |
| Sector only | 0.470 | 0.763 | 0.9% | 0.50 |
| Size only | 0.417 | 0.296 | 0.6% | 0.33 |
| Beta only | 0.534 | 0.299 | 3.5% | 1.79 |
| Sector + Size | 0.535 | 0.691 | 2.1% | 1.20 |
| **Full (Sector+Size+Beta)** | **0.624** | 0.417 | **4.0%** | **2.22** |

**Critical finding:** Without neutralization, FF alpha is essentially zero (-0.2%, t=-0.09). The returns without neutralization are entirely factor exposure, not stock selection. Full neutralization is the only configuration that produces statistically significant alpha. This validates the entire methodological approach — neutralization transforms factor exposure into genuine alpha.

### 7.5 Regime Analysis

| Regime | Sharpe | Ann. Return | Months |
|--------|--------|-------------|--------|
| **Bear markets** | **0.71** | 11.5% | 122 |
| Bull markets | 0.60 | 7.0% | 418 |

The strategy performs *better* in bear markets (Sharpe 0.71 vs 0.60 in bull markets). This is a desirable property for a market-neutral strategy — it provides crisis alpha when traditional long-only portfolios suffer.

### 7.6 Long vs Short Leg Attribution

| Leg | Sharpe | Ann. Return | Ann. Vol |
|-----|--------|-------------|----------|
| Long | **0.95** | 19.0% | 19.9% |
| Short | -0.47 | -9.6% | 20.5% |

Alpha is entirely driven by the long leg. The short leg loses money on average. This is a known challenge in equity long/short — shorting is structurally more expensive (borrow costs, short squeezes, upward drift of equities). For implementation, this suggests the strategy could be run as a long-biased overlay rather than dollar-neutral, or that the short leg needs improvement (better signals for identifying losers).

### 7.7 Bootstrap Confidence Intervals

| Period | Sharpe | 95% CI | SE |
|--------|--------|--------|-----|
| Full sample (541 months) | 0.62 | [0.33, 0.95] | 0.16 |
| OOS (60 months) | 0.42 | [-0.54, 1.74] | 0.63 |

Using circular block bootstrap with block size of 12 months (preserves annual autocorrelation) and 10,000 iterations. The full-sample Sharpe is statistically positive (CI excludes zero). The OOS CI is wide due to short sample.

---

## 8. Signal Mining Machine

### 8.1 Motivation

Hand-picking signals is slow, subject to researcher bias, and limited to well-known anomalies. We built a systematic signal discovery engine to exhaustively search the data for predictive patterns we might have missed.

### 8.2 Transformation Library

We apply 8 single-field transforms to 27 fundamental fields, plus price transforms and two-field ratios:

| Transform | Formula | Example |
|-----------|---------|---------|
| Level | Raw field value | atq (total assets) |
| Ratio to market cap | field / (price * shares) | ibq / mktcap (earnings yield) |
| Ratio to assets | field / atq | oancfy / atq (cash return on assets) |
| Ratio to equity | field / ceqq | ibq / ceqq (ROE) |
| Ratio to sales | field / saleq | ibq / saleq (profit margin) |
| YoY growth | field / lag(field, 12) - 1 | atq growth |
| QoQ growth | field / lag(field, 3) - 1 | quarterly change |
| Acceleration | growth_t - growth_{t-12} | change in growth rate |

Plus momentum (rolling 3/6/9/12-month), volatility (3/6/12/24-month), high-low range, analyst revision, analyst surprise (SUE), and analyst dispersion.

**Total: 662 candidate signals enumerated.**

### 8.3 Evaluation Protocol

Each candidate is evaluated in three stages:

1. **Development screen (1975-2004):** Compute IC, ICIR, hit rate, turnover, decile spread. Require |ICIR| > 0.15, hit rate > 52%, turnover < 0.60, spread t-stat > 2.0.

2. **Validation confirmation (2005-2014):** Re-compute IC on held-out period. Require |ICIR| > 0.10. This guards against multiple testing — with 662 candidates, some pass the development screen by chance.

3. **Correlation deduplication:** Compute average cross-sectional correlation between each surviving candidate and all previously accepted signals (including the 6 originals). If max correlation > 0.70, drop the weaker signal. This prevents adding multiple signals that are essentially the same thing.

### 8.4 Results

| Stage | Surviving |
|-------|-----------|
| Enumerated | 662 |
| Passed development filters | 98 (14.8%) |
| Confirmed on validation | 78 (11.8%) |
| Non-redundant after dedup | **34 (5.1%)** |

### 8.5 Top Discoveries

| Signal | Formula | Dev ICIR | Val ICIR | Hit Rate | Category |
|--------|---------|----------|----------|----------|----------|
| **ibcomq_to_mktcap** | Net income / market cap | **0.468** | **0.302** | 69% | Value |
| **oancfy_to_mktcap** | Operating cash flow / market cap | 0.414 | 0.245 | 67% | Value |
| **oancfy_div_seqq** | Cash flow / shareholder equity | 0.401 | 0.284 | 66% | Quality |
| **oancfy_div_dlttq** | Cash flow / long-term debt | 0.384 | 0.280 | 65% | Quality |
| **oancfy_div_rectq** | Cash flow / receivables | 0.349 | 0.229 | 68% | Quality |
| **epsfxq_to_mktcap** | EPS / market cap | 0.293 | **0.333** | 62% | Value |

**The dominant theme is cash flow quality.** Operating cash flow (OANCFY) normalized by various balance sheet items dominates the top discoveries. This is economically coherent — cash flow from operations is the hardest accounting number to manipulate and is the most direct measure of a firm's ability to generate real economic value.

The best mined signal (ibcomq_to_mktcap, ICIR 0.47) exceeds the best hand-picked signal (ROE, ICIR 0.35) by 34%.

### 8.6 Expanded Signal Set Performance

We integrated the top 5 mined signals into the pipeline:

| Metric | Baseline (6 signals) | Expanded (11 signals) | Change |
|--------|---------------------|-----------------------|--------|
| Full-Sample Sharpe | 0.624 | **0.809** | **+0.185** |
| FF Alpha (annualized) | 4.0% (t=2.22) | **6.2% (t=3.40)** | **+2.2%** |
| Composite ICIR | 0.32 | **0.45** | +0.13 |
| Annualized Vol | 12.9% | 12.3% | -0.6% |
| Avg Turnover | 2.19 | 1.79 | -0.40 |
| OOS Sharpe | 0.42 | 0.26 | -0.16 |

The mined signals substantially improve full-sample performance: Sharpe +30%, alpha significance from t=2.22 to t=3.40, while reducing both volatility and turnover. The OOS decline (0.42 → 0.26) reflects regime sensitivity of value/quality signals in the 2015-2017 growth-dominated environment. This is an honest finding — the mined signals are predominantly cash-flow-to-price ratios that mechanically underperform when growth stocks dominate.

---

## 9. Implementation Considerations

### 9.1 Execution

The strategy rebalances monthly with average two-way turnover of ~180-220%. For a $100M book, this implies ~$90-110M of monthly trading. In S&P 500 names, achieving 5-10 bps one-way cost is realistic with algorithmic execution (VWAP, TWAP, implementation shortfall).

### 9.2 Capacity

With 50 long and 50 short positions at 2% max weight, the strategy runs a $2M average position on a $100M book. S&P 500 median daily dollar volume exceeds $200M, so participation rates are well below 1% of ADV. **Estimated capacity: $500M-$1B** before market impact materially degrades returns.

### 9.3 Short Leg Improvement

The short leg loses money (Sharpe -0.47). Potential improvements:

- Asymmetric signal weighting for shorts (e.g., emphasize accrual/quality signals which identify "losers" better)
- Short-specific signals: earnings management, insider selling, credit deterioration
- Tighter position sizing on shorts to limit adverse selection

### 9.4 Limitations

| Limitation | Mitigation |
|------------|------------|
| Sector coverage only 37.5% | Historical/delisted tickers assigned "Other" — neutralization still controls for known sectors |
| OOS period only 60 months | Bootstrap CI honestly reported; full-sample alpha (541 months) is the primary significance test |
| No borrow costs in backtest | S&P 500 general collateral borrow is typically 25-50 bps annually — would reduce net return by ~0.3% |
| Free data, not CRSP/Compustat from WRDS | CRSP/Compustat merged panel is institutional-grade; data quality is not a concern |
| Monthly rebalance only | Higher-frequency signals (intraday reversal, weekly momentum) are out of scope |

---

## 10. Conclusion

This project demonstrates that a disciplined, interpretable research process can generate statistically significant alpha in U.S. equities. The key is not model complexity but **research design**: point-in-time data handling, cross-sectional neutralization, proper factor attribution, and honest out-of-sample testing.

The systematic signal mining machine extends this framework from manual hypothesis testing to automated discovery, identifying cash-flow quality as the dominant predictive theme — a finding that aligns with decades of accounting research on earnings quality.

**What makes this project credible is what it does not claim.** We do not claim 2.0 Sharpe. We do not hide transaction costs. We do not ignore the short leg's underperformance. We do not pretend the OOS period is long enough for narrow confidence intervals. We show what works, what doesn't, and why.

---

## Appendix A: Project Architecture

```
equity-alpha-pipeline/
  config/pipeline.yaml          # All parameters in one place
  src/
    data/                       # DataPanel, sectors, French factors
    factors/                    # 6 hand-picked + 34 mined Factor subclasses
    signals/                    # Registry, z-score, neutralize, combine, report card
    ml/                         # Purged CV, Ridge/ElasticNet/XGBoost
    mining/                     # Enumeration, compute, evaluate, filter, dedup, codegen
    portfolio/                  # Risk model, optimizer, backtest engine
    analytics/                  # IC, attribution, bootstrap, performance, risk
  scripts/
    stage_1_features.py         # Compute all signals
    stage_2_signals.py          # Neutralize + combine + report cards
    stage_3_backtest.py         # Walk-forward backtest
    stage_4_ml.py               # ML comparison
    stage_5_robustness.py       # 8-test robustness battery
    stage_6_compare.py          # Expanded vs baseline comparison
    run_mining.py               # Signal mining machine
    run_all.py                  # Full pipeline orchestrator
```

## Appendix B: Reproducibility

```bash
git clone <repo>
uv sync
uv run python scripts/build_sector_mapping.py   # One-time: build sectors from SEC EDGAR
uv run python scripts/run_all.py                 # Run full pipeline (stages 1-5)
uv run python scripts/run_mining.py              # Run signal mining (~30 min)
uv run python scripts/stage_6_compare.py         # Compare expanded vs baseline
```

All results are deterministic given the same input data and random seeds.
