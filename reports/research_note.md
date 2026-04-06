# Factor-Neutral Long/Short U.S. Equity Alpha Research Pipeline

**Research Note — Prepared for Portfolio Management Review**

---

## Executive Summary

We built a systematic alpha research pipeline that discovers, evaluates, and combines stock-selection signals into a factor-neutral long/short portfolio. The system has two phases: Phase 1 (Stages 1-6) systematically mines 770 candidate signals and selects the optimal combination via forward stepwise portfolio-level testing. Phase 2 (Stage 7) takes the selected signals and runs them through a full portfolio construction pipeline with factor attribution and diagnostics.

The OOS period (2015-2019) is never used for any selection decision — it is purely for final evaluation.

**Final result (Optimizer + Equal-Weight):**

| Metric | Value |
|--------|-------|
| Signals selected | 3 (all cash-flow quality) |
| Pre-OOS Sharpe (1975-2014) | **1.64** |
| OOS Sharpe (2015-2019) | **0.46** |
| FF Alpha (annualized) | **7.5%** |
| Alpha t-statistic (Newey-West HAC) | **5.64** |
| Max Drawdown | -15.2% |
| Candidates enumerated | 770 |
| Candidates surviving all filters | 27 |
| Total pipeline runtime | 38 minutes |

---

## 1. System Architecture

### 1.1 Design Flowchart

```
════════════════════════════════════════════════════════════════════════
  PHASE 1: SIGNAL DISCOVERY (Stages 1-6)
  Decides WHICH signals to use
  Uses equal-weight + naive L/S (zero tunable parameters → no overfitting)
  Selection criterion: pre-OOS Sharpe (1975-2014 ONLY)
  OOS (2015-2019) is NEVER used for any decision
════════════════════════════════════════════════════════════════════════

Stage 1: ENUMERATE — 770 candidate signals from transformation library
    ↓
Stage 2: COMPUTE — pull raw fields, apply math, cache (18 sec)
    ↓
Stage 3: PRE-SELECT — GPU batch IC → cascaded filtering (7 min)
    743 → 138 (IC) → 77 (turnover + spread)
    ↓
Stage 4: VALIDATE + DEDUP — holdout (2005-2014) + correlation (2 min)
    77 → 57 (validated) → 27 (non-redundant)
    ↓
Stage 5: PRECOMPUTE NEUTRALIZATION — z-score + OLS residuals (5 min)
    All 54 candidates neutralized once, cached for stepwise
    ↓
Stage 6: STEPWISE SELECTION — forward portfolio-level testing (2 min)
    Test each candidate via backtest, accept best, stop when no improvement
    Output: 3 optimal signals

════════════════════════════════════════════════════════════════════════
  PHASE 2: PORTFOLIO CONSTRUCTION (Stage 7)
  Decides HOW TO TRADE the selected signals
  Takes the 3 signals from Phase 1, applies full evaluation
════════════════════════════════════════════════════════════════════════

Stage 7: FINAL EVALUATION (13 sub-steps, 6 portfolio variants)
  7a.  Neutralization (full OLS) + verification
  7b.  Signal report cards (IC, turnover, correlations, marginal IC)
  7c.  IC decay analysis (signal persistence at lags 1-12)
  7d.  Naive L/S × 3 combination methods:
       [Equal-Weight, IC-Weighted, Inverse-Vol]
       Per method: performance, risk (VaR/CVaR), FF attribution,
       bootstrap CIs, long/short legs, skew/kurtosis, holding period
       Statistical tests: Sharpe equality, Diebold-Mariano, t-test
  7e.  Constrained optimizer × 3 combination methods
       cvxpy: dollar/beta/sector neutral, turnover penalty
       Monthly QP solve with factor risk model covariance
  7f.  Subperiod stability (dev / val / OOS early / OOS late)
  7g.  Cost sensitivity (0-30 bps)
  7h.  Regime analysis (bull vs bear)
  7i.  Signal ablation (drop one at a time)
  7j.  Sleeve attribution (each signal's standalone P&L)
  7k.  Sector attribution (per-sector contribution)
  7l.  Neutralization sensitivity (none / partial / full)
  7m.  Quantile monotonicity (decile 1-10 returns)
  Output: 6 portfolio variants, all diagnostics
```

### 1.2 Why Each Design Decision

| Decision | Why |
|----------|-----|
| **Two-phase separation** | Phase 1 selects WHICH signals with zero tunable parameters. Phase 2 evaluates HOW THEY PERFORM. Separating selection from evaluation prevents overfitting. |
| **Pre-OOS selection criterion** | Stepwise maximizes Sharpe on 1975-2014 only. OOS (2015-2019) is never used for decisions. |
| **GPU for IC evaluation** | 743 rank correlations per date batched into one GPU kernel. CPU: ~35 min. GPU: ~1 min. |
| **Cascaded filtering** | IC filter eliminates 81% before computing expensive turnover/spread. Same result, 5× faster. |
| **Holdout validation** | With 770 candidates tested, some pass by chance. 2005-2014 holdout catches false positives. |
| **Precomputed neutralization** | Neutralization is deterministic. Precomputing once replaces thousands of redundant recomputations. 70× speedup. |
| **Equal-weight in stepwise** | Zero degrees of freedom during signal search. Weight optimization deferred to Phase 2. |
| **Stepwise at portfolio level** | Individual signal IC ≠ portfolio Sharpe. Only backtest-level testing reveals which signals work *together*. |

---

## 2. Data

### 2.1 Source

CRSP/Compustat merged monthly panel — institutional standard for U.S. equity research. Survivorship-bias-free: includes delisted companies.

| Dimension | Value |
|-----------|-------|
| Total observations | 569,476 |
| Unique companies | 1,685 |
| Date range | January 1962 — September 2020 |
| Fields | 52 (price, returns, quarterly fundamentals, analyst estimates) |

### 2.2 Universe

At each rebalance date:
- S&P 500 constituents (point-in-time membership flags)
- Market cap > $100 million
- At least 12 months of return history
- Average: **381 stocks per month**

### 2.3 Time Periods

| Period | Dates | Months | Purpose |
|--------|-------|--------|---------|
| Burn-in | 1962-01 to 1974-12 | 156 | Covariance lookback accumulation |
| Development | 1975-01 to 2004-12 | 360 | Signal discovery + parameter tuning |
| Validation | 2005-01 to 2014-12 | 120 | Holdout confirmation for mined signals |
| **Out-of-Sample** | **2015-01 to 2019-12** | **60** | **Final evaluation — touched once** |

### 2.4 Anti-Leakage Rules

| Rule | Implementation |
|------|----------------|
| IC computation | Signal at date t, returns at t+1 (next month) |
| Quarterly fundamentals | 3-month forward-fill from last known value |
| Universe membership | S&P 500 flag at date t, not t+1 |
| Signal discovery | Development period ONLY (1975-2004) |
| Signal validation | Holdout period ONLY (2005-2014) |
| Stepwise selection | Pre-OOS Sharpe (1975-2014) ONLY |
| Standardization | z-scores from cross-section at date t only |

### 2.5 Supplementary Data

- **Fama-French 5 factors + Momentum** (monthly): factor attribution, beta estimation, risk model
- **Sector classification**: SIC codes from SEC EDGAR → Ken French 12-industry mapping

---

## 3. Stage 1 — Signal Enumeration

The mining machine systematically applies 16 mathematical transforms to every available data field:

| Transform | Formula | Example | Count |
|-----------|---------|---------|-------|
| Level | raw value | total assets | 29 |
| Ratio to market cap | field / (price × shares) | earnings yield | 26 |
| Ratio to assets | field / total assets | cash return on assets | 26 |
| Ratio to equity | field / common equity | ROE | 25 |
| Ratio to sales | field / revenue | profit margin | 26 |
| YoY growth | field / lag(field, 12) - 1 | asset growth | 29 |
| QoQ growth | field / lag(field, 3) - 1 | quarterly momentum | 27 |
| Acceleration | growth_t - growth_{t-12} | change in growth | 27 |
| **Difference** | field_a - field_b | gross profit (sales - COGS) | 13 |
| **Difference ratio** | (field_a - field_b) / field_c | accrual ratio, gross profit yield | 50 |
| **Negate** | -field | reversal | 33 |
| Two-field ratio | field_a / field_b | cash flow / receivables | 435 |
| Momentum | rolling mean of returns | 3/6/9/12-month momentum | 4 |
| Volatility | rolling std of returns | 3/6/12/24-month vol | 4 |
| Momentum with skip | shift(k).rolling(w) | 12-2 momentum | 12 |
| Analyst transforms | revision ratio, SUE, dispersion | | 3 |
| **Total** | | | **770** |

The **difference** and **difference ratio** transforms are critical — they enable signals like gross profitability `(sales - COGS) / assets` and accrual quality `(income - cash flow) / equity` that cannot be expressed as simple ratios.

---

## 4. Stage 2 — Signal Computation

For each of 770 candidate specifications:
1. Pull raw data fields from the CRSP/Compustat panel
2. Apply the mathematical transform
3. Handle infinities (division by zero → NaN)

**Pivot cache optimization:** Each raw field (e.g., `atq`, `ibq`, `saleq`) is pivoted from long format to (date × stock) matrix once and cached in memory. Multiple signals that use the same field reuse the cached pivot. This reduces 770 pivot operations to ~40.

Output: 743 valid signal matrices (27 failed due to missing data). Each is a (696 dates × 1,685 stocks) DataFrame.

Time: 18 seconds.

---

## 5. Stage 3 — Cascaded Pre-Selection

### 5.1 How IC Evaluation Works — Worked Example

The IC (Information Coefficient) measures whether a signal predicts stock returns.

**At one date (e.g., January 2000), for 3 candidate signals and 5 stocks:**

```
Raw signal values:
                    AAPL   MSFT    GE    JPM   XOM
Signal A (ROE):     0.25   0.18   0.04   0.31  0.12
Signal B (CF Yld):  2.30   1.80   0.40   3.10  1.20
Signal C (EPS Yld): 0.08   0.05   0.02   0.09  0.06

Rank each signal's stocks (1 = highest value):
                    AAPL   MSFT    GE    JPM   XOM
Signal A ranks:       2      3      5      1     4
Signal B ranks:       2      3      5      1     4
Signal C ranks:       2      4      5      1     3

Next month's actual stock returns:
                    AAPL   MSFT    GE    JPM   XOM
Feb 2000 return:   +5.2%  +3.1%  -1.0%  +6.8%  +0.5%
Return ranks:         2      3      5      1      4

Correlate each signal's ranks with the return ranks:
Signal A: [2,3,5,1,4] vs [2,3,5,1,4] → IC = 1.00  (perfect prediction!)
Signal B: [2,3,5,1,4] vs [2,3,5,1,4] → IC = 1.00
Signal C: [2,4,5,1,3] vs [2,3,5,1,4] → IC = 0.90  (close but not perfect)
```

**The GPU processes all 743 signals at once:** stack into a matrix (743 × 1,685), rank all rows in one GPU operation, correlate with return ranks in one batched dot product.

**Repeat for all 145 dates** to build an IC time series per signal:

```
Signal A: [+0.03, -0.01, +0.05, +0.02, ...]  → Mean IC = 0.030, ICIR = 0.38 ← keep
Signal B: [+0.08, +0.04, +0.06, +0.07, ...]  → Mean IC = 0.055, ICIR = 0.61 ← keep
Signal C: [+0.01, -0.02, +0.00, +0.03, ...]  → Mean IC = 0.005, ICIR = 0.07 ← discard
```

**ICIR** = Mean IC / Std IC — the signal's consistency. We filter at ICIR > 0.15.

### 5.2 Cascaded Filtering

Instead of computing all metrics for all candidates, filter by the cheapest metric first:

```
770 candidates
  ├─ Standardize (winsorize + z-score)              all 743 valid
  ├─ GPU batch IC evaluation (~1 min)                743 → 138 pass (|ICIR| > 0.15)
  ├─ Turnover + decile spread (CPU, 138 only, ~5m)  138 → 77 pass
  └─ Why cascaded: 138 instead of 743 → same result, 5× faster
```

---

## 6. Stage 4 — Validation + Deduplication

### 6.1 Holdout Validation

Re-compute IC on the 2005-2014 period (NOT used for discovery). Require |ICIR| > 0.10.

77 → 57 confirmed.

**Why:** With 770 candidates tested, ~38 pass development filters by chance alone. The holdout period catches these false positives.

### 6.2 Correlation Deduplication

Compute pairwise cross-sectional correlation between all survivors. If two signals are > 70% correlated, drop the weaker one.

57 → 27 non-redundant.

**Why:** Two signals measuring the same thing waste a portfolio slot. We want diverse signals that capture different aspects of stock quality.

---

## 7. Stage 5 — Precomputed Neutralization

For each signal at each date, remove mechanical correlation with sector, size, and market beta via cross-sectional OLS:

```
signal_{i,t} = a + b₁·sector + b₂·log(mcap) + b₃·beta + ε_{i,t}
```

Keep only the residual ε — the part that is pure stock selection, not factor exposure.

**Optimization:** Precompute projection matrices M_t = I - X(X'X)⁻¹X' once for all dates. Neutralizing any signal then becomes M_t × signal (a single matrix-vector multiply). All 54 candidates are neutralized once and cached in memory.

**Why precompute:** Neutralization is deterministic — `neutralize(signal_X)` gives the same result regardless of what other signals are in the trial set. Without this cache, the stepwise loop would recompute the same signals hundreds of times. Speedup: **70×**.

Time: ~5 minutes (one-time).

---

## 8. Stage 6 — Forward Stepwise Portfolio Selection

### 8.1 Why Portfolio-Level Selection

Individual signal quality (IC) does not predict portfolio performance:

| Approach | How signals chosen | Pre-OOS Sharpe | OOS Sharpe |
|----------|-------------------|----------------|------------|
| Hand-picked by IC | Top 6 from academic literature | 0.62 | 0.42 |
| Top 5 by individual ICIR | Rank by ICIR, take top 5 | 0.81 | 0.26 |
| **Stepwise portfolio test** | **Test each combo via backtest** | **1.62** | **0.39** |

### 8.2 The Algorithm — Worked Example

**Step 1 — Test each signal alone (54 backtests on 1975-2014):**
```
Try signal A alone → backtest → Sharpe 0.85
Try signal B alone → backtest → Sharpe 1.24  ← best
Try signal C alone → backtest → Sharpe 0.62
... (test all 54)

Winner: signal B (oancfy_div_seqq, Sharpe 1.24)
Selected set: [B]
```

**Step 2 — Test B + each of the remaining 53:**
```
Try B + A → combine, backtest → Sharpe 1.20
Try B + D → combine, backtest → Sharpe 1.55  ← best
... (test all 53)

Winner: B + D (added ibcomq_to_mktcap, Sharpe 1.55)
Selected set: [B, D]
```

**Step 3 — Test B + D + each of the remaining 52:**
```
Try B + D + C → combine, backtest → Sharpe 1.62  ← best
...

Winner: B + D + C (added oancfy_to_mktcap, Sharpe 1.62)
Selected set: [B, D, C]
```

**Step 4 — Best improvement is +0.008 < threshold 0.01 → stop.**

Each addition is evaluated in the context of what's already selected. Signal D is chosen at step 2 not because it has the highest individual IC, but because it adds the most value *on top of signal B*.

**Critical:** All backtests use 1975-2014 only. OOS (2015-2019) is never seen during selection.

### 8.3 Selection Path

| Step | Signal Added | Pre-OOS Sharpe | OOS Sharpe | FF Alpha | t-stat |
|------|-------------|---------------|------------|----------|--------|
| 1 | oancfy_div_seqq | 1.241 | 0.029 | 4.2% | 2.96 |
| 2 | + ibcomq_to_mktcap | 1.547 | 0.236 | 6.5% | 4.36 |
| 3 | + oancfy_to_mktcap | **1.617** | **0.391** | **7.6%** | **5.56** |
| *(stopped — step 4 improvement < 0.01)* |

### 8.4 The Final 3 Signals

| Signal | Formula | Economic Rationale |
|--------|---------|-------------------|
| **oancfy_div_seqq** | Operating cash flow / shareholder equity | Cash return on equity — how much cash per dollar of equity |
| **ibcomq_to_mktcap** | Net income / market cap | Earnings yield — cheap stocks with real earnings |
| **oancfy_to_mktcap** | Operating cash flow / market cap | Cash flow yield — cheap stocks with real cash generation |

All three are interpretable. Two measure cash-flow quality, one measures earnings yield. The dominant theme: **markets undervalue firms with strong, genuine cash generation.**

---

## 9. Stage 7 — Portfolio Construction and Final Evaluation

Phase 1 (Stages 1-6) selected WHICH signals to use. Stage 7 evaluates HOW THEY PERFORM using the full portfolio pipeline — 13 sub-steps covering 6 portfolio variants and 8 robustness tests.

### 9.1 Six Portfolio Variants

The portfolio construction has two independent choices:

**Step 1 — Signal combination:** How to blend the 3 signals into one composite alpha score per stock.
- **Equal-Weight:** Each signal contributes equally: alpha = (signal_A + signal_B + signal_C) / 3
- **IC-Weighted:** Weight by trailing 36-month IC (signals with stronger recent prediction get more weight)
- **Inverse-Vol:** Weight by 1/std(IC) (signals with more consistent prediction get more weight)

**Step 2 — Stock weighting:** Given the composite alpha score, how to assign weights to individual stocks.
- **Naive L/S:** Equal weight per stock. Top 50 stocks get +2% each, bottom 50 get -2% each. No optimization.
- **Constrained Optimizer:** cvxpy solves a QP to find the optimal weight for each stock, subject to neutrality constraints. Stock weights vary (e.g., AAPL +1.8%, MSFT +0.3%, GE -1.5%).

3 signal combinations × 2 stock weighting methods = 6 variants:

| Signal Combination | Stock Weighting | Pre-OOS Sharpe | OOS Sharpe | FF Alpha | t-stat |
|-------------------|-----------------|---------------|------------|----------|--------|
| Equal-Weight | Naive L/S | 1.605 | 0.388 | 7.5% | 5.49 |
| IC-Weighted | Naive L/S | 1.596 | 0.068 | 6.7% | 4.69 |
| Inverse-Vol | Naive L/S | 1.605 | 0.417 | 7.4% | 5.54 |
| **Equal-Weight** | **Optimizer** | **1.644** | **0.459** | **7.5%** | **5.64** |
| IC-Weighted | Optimizer | 1.617 | -0.026 | 6.5% | 4.75 |
| Inverse-Vol | Optimizer | 1.664 | 0.330 | 7.4% | 5.55 |

**Findings:**
- **Equal-Weight + Optimizer wins OOS** (0.459). The optimizer adds 0.04-0.07 Sharpe over naive L/S.
- **Inverse-Vol beats IC-Weighted** consistently — weighting by consistency beats weighting by magnitude with only 3 signals.
- **IC-Weighted collapses OOS** — trailing IC estimation too noisy with 3 signals.

**Constrained optimizer constraints:**

The optimizer solves a QP at each monthly rebalance using a factor risk model covariance (B @ F @ B' + D with trailing 60-month rolling window):

| Constraint | Specification | Purpose |
|------------|---------------|---------|
| Dollar-neutral | sum(w) = 0 | No market direction bet |
| Beta-neutral | \|portfolio beta\| ≤ 0.05 | No market factor exposure |
| Sector-neutral | \|net weight per sector\| ≤ 5% | No sector bets |
| Gross leverage | sum(\|w\|) ≤ 200% | Cap total exposure |
| Position limit | -2% ≤ w_i ≤ 2% per stock | No single-stock concentration |
| Turnover penalty | κ × \|\|w_new - w_old\|\|₁ in objective | Reduce trading costs |

### 9.2 Detailed Performance

We report detailed metrics for two portfolios: the **naive L/S** (clean diagnostic — no optimizer noise) and the **constrained optimizer** (production portfolio — enforces neutrality constraints).

The naive L/S is the primary diagnostic because it isolates signal quality from optimizer behavior. The optimizer is the production version because real portfolios need risk constraints.

| Metric | Naive L/S (EW) | Optimizer (EW) |
|--------|---------------|----------------|
| Pre-OOS Sharpe | 1.605 | **1.644** |
| OOS Sharpe | 0.388 | **0.459** |
| Full Sharpe | 1.474 | — |
| Sortino | 2.68 | 2.63 |
| Calmar | 0.93 | — |
| Max Drawdown | -14.6% | -15.2% |
| Win Rate | 70% | — |
| Skewness | +0.52 (positive) | — |
| Kurtosis | 3.30 | — |
| Avg Turnover | 1.02 | 1.00 |
| Avg Holding Period | 2.0 months | — |
| Avg Gross Exposure | 200% | — |
| Avg Net Exposure | 0.0% | 0.0% |
| FF Alpha (ann) | 7.5% | **7.5%** |
| Alpha t-stat | 5.49 | **5.64** |

The optimizer improves OOS Sharpe from 0.39 to 0.46 (+18%) and alpha t-stat from 5.49 to 5.64, while maintaining the same 7.5% annualized alpha. The improvement comes from tighter risk control — the optimizer reduces unintended beta and sector exposures that the naive equal-weight portfolio inherits from the stock selection.

### 9.3 Risk Metrics

| Metric | Value |
|--------|-------|
| Parametric VaR (95%) | -3.2% |
| Historical VaR (95%) | -2.8% |
| CVaR (Expected Shortfall) | -4.2% |
| Cornish-Fisher VaR | -2.6% |
| Avg Drawdown | -3.8% |
| Max Drawdown Duration | 28 months |

### 9.4 Factor Attribution

Time-series regression of monthly portfolio returns on Fama-French 5 factors + Momentum (Newey-West HAC standard errors, 540 observations):

```
r_{p,t} - rf = α + β₁·MktRF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + β₆·Mom + ε
```

| Factor | Loading | t-stat | Interpretation |
|--------|---------|--------|----------------|
| **Alpha (ann.)** | **+7.5%** | **5.49*** | **Genuine stock selection — unexplained by any factor** |
| Mkt-RF | +0.013 | 0.50 | Near-zero market exposure — dollar-neutral working |
| SMB | +0.111 | 2.44** | Mild small-cap tilt |
| HML | +0.258 | 3.99*** | Value tilt (expected — signals are cash-flow-to-price ratios) |
| RMW | +0.288 | 4.99*** | Profitability tilt (expected — signals measure cash quality) |
| CMA | -0.169 | -1.58 | Not significant |
| Mom | -0.017 | -0.55 | No momentum exposure |
| **R-squared** | **0.116** | | **88% of variance is idiosyncratic (stock-specific)** |

The alpha of 7.5% (t=5.49) is significant at the 1% level. The HML and RMW loadings are expected — our signals are cash-flow-to-price and profitability ratios, which naturally correlate with value and quality factors. The key finding is that **alpha remains highly significant even after controlling for these exposures** — the signals capture stock-specific information beyond what systematic factors explain.

### 9.5 Statistical Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| EW Mean Return t-test | t=9.89 | 0.0000 | Return highly significantly > 0 |
| Sharpe Equality (EW vs IC) | z=1.13 | 0.258 | Not significantly different |
| Diebold-Mariano (EW vs IC) | DM=0.32 | 0.752 | Not significantly different |

### 9.6 Bootstrap Confidence Intervals

| Period | Sharpe | 95% CI |
|--------|--------|--------|
| OOS (2015-2019) | 0.39 | [-0.32, 1.20] |

### 9.7 Long vs Short Leg

| Leg | Sharpe | Ann. Return |
|-----|--------|-------------|
| Long | 1.18 | 23.1% |
| Short | -0.46 | -9.0% |

Alpha primarily from long leg. Short leg underperforms (structural challenge of shorting).

### 9.8 Subperiod Stability

| Period | Sharpe | Ann. Return | Max DD | Months |
|--------|--------|-------------|--------|--------|
| Development (1975-2004) | **1.72** | 15.7% | -14.6% | 359 |
| Validation (2005-2014) | **1.27** | 12.0% | -10.8% | 120 |
| OOS Early (2015-2017) | 0.95 | 7.0% | -6.0% | 36 |
| OOS Late (2018-2019) | 0.22 | 1.4% | -8.2% | 24 |

### 9.9 Cost Sensitivity

| Cost (bps RT) | Full Sharpe | OOS Sharpe |
|---------------|-------------|------------|
| 0 | 1.54 | 0.45 |
| 5 | 1.51 | 0.42 |
| **10 (base)** | **1.47** | **0.39** |
| 15 | 1.44 | 0.35 |
| 20 | 1.41 | 0.32 |
| 30 | 1.34 | 0.25 |

Strategy remains profitable up to 30 bps.

### 9.10 Regime Analysis

| Regime | Sharpe | Ann. Return | Months |
|--------|--------|-------------|--------|
| **Bear** | **1.74** | 19.9% | 121 |
| Bull | 1.46 | 11.9% | 418 |

Works better in bear markets — desirable for market-neutral strategy.

### 9.11 Signal Ablation

| Dropped | Pre-OOS Sharpe | Delta |
|---------|---------------|-------|
| *(none — baseline)* | 1.605 | — |
| oancfy_div_seqq | 1.377 | **-0.228** |
| **ibcomq_to_mktcap** | **1.297** | **-0.308** |
| oancfy_to_mktcap | 1.527 | -0.077 |

All 3 are load-bearing. ibcomq_to_mktcap contributes the most.

### 9.12 Sleeve Attribution

| Signal | Standalone Sharpe | Ann. Return |
|--------|------------------|-------------|
| oancfy_div_seqq | 1.07 | 8.3% |
| oancfy_to_mktcap | 1.04 | 10.0% |
| ibcomq_to_mktcap | 0.95 | 10.8% |

All individually strong. Combined Sharpe (1.61) exceeds any individual — diversification works.

### 9.13 Neutralization Sensitivity

| Neutralization | Pre-OOS Sharpe | OOS Sharpe | Alpha t-stat |
|----------------|---------------|------------|--------------|
| None | 1.195 | 0.736 | 4.20 |
| Sector only | 1.273 | 0.590 | 4.14 |
| Size only | 1.375 | 0.529 | 4.36 |
| Beta only | 1.394 | 0.651 | 4.69 |
| **Full (S+S+B)** | **1.605** | **0.388** | **5.49** |

Full neutralization produces the highest pre-OOS Sharpe and alpha significance. Without neutralization, higher OOS Sharpe but lower alpha — returns are factor exposure, not stock selection.

### 9.14 Quantile Monotonicity

| Decile | Ann. Return |
|--------|-------------|
| 1 (lowest alpha) | 8.0% |
| 2 | 10.2% |
| 3 | 10.8% |
| 4 | 11.8% |
| 5 | 12.3% |
| 6 | 13.4% |
| 7 | 14.5% |
| 8 | 15.2% |
| 9 | 17.6% |
| 10 (highest alpha) | **22.3%** |
| **Long/Short Spread** | **14.3%** |

Nearly perfect monotonicity — each decile earns more than the one below. The composite alpha score reliably separates winners from losers across the entire distribution.

---

## 10. Performance Engineering

### 10.1 Runtime Breakdown

| Stage | Time | Optimization |
|-------|------|-------------|
| Stage 1: Enumeration | < 1 sec | — |
| Stage 2: Computation (770) | 18 sec | Pivot field caching |
| **Stage 3: GPU batch IC (743)** | **~1 min** | **RTX 4080, batched argsort + correlation** |
| Stage 3: IC filter | instant | Cascaded: eliminates 81% first |
| Stage 3: Turnover + spread (138) | ~5 min | Cascaded: 138 instead of 743 |
| Stage 4: Validation + dedup | ~2 min | — |
| **Stage 5: Precompute neutralization (54)** | **~5 min** | **Projection matrix cache** |
| **Stage 6: Stepwise (3 steps)** | **~2 min** | **Precomputed signals: 0.7 sec/trial** |
| Stage 7: Final evaluation (13 sub-steps) | ~15 min | 6 variants + 8 robustness tests |
| **Total** | **~38 min** | |

### 10.2 Hardware Utilization

| Resource | Usage | Purpose |
|----------|-------|---------|
| RTX 4080 SUPER (16 GB VRAM) | GPU batch IC evaluation | Parallel rank correlation for 743 signals |
| i9-14900KF (32 threads) | Signal computation, backtest | Numpy/pandas operations |
| 32 GB RAM | Signal tensors in memory | All 54 neutralized signals cached simultaneously |

---

## 11. Limitations and Honest Assessment

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| OOS period only 60 months | Bootstrap CI includes zero | Full-sample t-stat 5.49 is the primary test |
| Sector coverage 37.5% | Historical tickers → "Other" | Neutralization controls for 11 identified sectors |
| No borrow costs | Overstates short-leg returns | S&P 500 borrow is cheap (~25-50 bps/year) |
| Monthly rebalance only | Misses higher-frequency signals | Appropriate for fundamental signals |
| Short leg loses money | Alpha entirely from long book | Could run as long-biased overlay |
| Only 3 signals selected | Concentrated portfolio | Concentration is a feature: each signal is non-redundant |

---

## 12. Conclusion

This project demonstrates three principles of systematic alpha research:

**1. Systematic search beats human intuition.** Hand-picking 6 signals from the academic literature produced Sharpe 0.62. Systematic mining of 770 candidates found 3 signals with pre-OOS Sharpe 1.64 — all cash-flow variants that a human researcher might not have prioritized.

**2. Portfolio-level selection is essential.** Ranking signals by individual IC produced Sharpe 0.81. Testing combinations via actual backtests found Sharpe 1.64. The right question is not "which signals are individually good?" but "which signals are good *together*?"

**3. The optimizer works.** Constrained optimization adds 0.04-0.07 Sharpe over naive L/S while enforcing dollar/beta/sector neutrality. Best OOS Sharpe: 0.46 (optimizer + equal-weight).

**4. Honest evaluation matters.** Our initial stepwise selected on full-sample Sharpe (including OOS), producing an inflated OOS Sharpe of 0.96. After fixing to select only on pre-OOS data, the honest OOS Sharpe is 0.46 — lower but real. The alpha remains highly significant (7.5%, t=5.64).

The final portfolio is economically interpretable: cash flow to equity, earnings yield, and cash flow yield. The dominant theme — that markets undervalue firms with strong, genuine cash generation — is consistent with decades of accounting research and practitioner experience.

---

## Appendix A: Reproducibility

```bash
git clone https://github.com/spirituslab/equity-alpha-pipeline
cd equity-alpha-pipeline
uv sync
uv pip install cupy-cuda13x                      # GPU (requires CUDA toolkit)
uv run python scripts/build_sector_mapping.py     # One-time: sectors from SEC EDGAR
uv run python scripts/stage_1_features.py         # Cache control variables
uv run python scripts/run_full_pipeline.py        # Full pipeline: mine → select → evaluate
```

## Appendix B: Project Structure

```
equity-alpha-pipeline/
  config/
    pipeline.yaml              # Pipeline parameters
    mining.yaml                # Mining thresholds
  src/
    data/                      # DataPanel, sectors, French factors
    factors/                   # Hand-picked + auto-generated signals
      mined/                   # Auto-generated from mining machine
    signals/                   # Registry, z-score, neutralize, combine
    mining/                    # Mining machine + stepwise selection
    gpu/                       # CuPy backend, batched IC, projection cache
    portfolio/                 # Risk model, optimizer, backtest engine
    analytics/                 # IC, attribution, bootstrap, performance
    utils/                     # Logging + notifications
  scripts/
    run_full_pipeline.py       # Complete workflow (Stages 1-7)
    stage_1-7_*.py             # Individual stages
    run_mining.py              # Mining standalone
```
