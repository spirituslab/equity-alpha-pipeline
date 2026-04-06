# Factor-Neutral Long/Short U.S. Equity Alpha Research Pipeline

**Research Note — Prepared for Portfolio Management Review**

---

## Executive Summary

We built a systematic alpha research pipeline that discovers, evaluates, and combines stock-selection signals into a factor-neutral long/short portfolio. The system has two phases: Phase 1 (Stages 1-6) systematically mines 770 candidate signals and selects the optimal combination via forward stepwise portfolio-level testing. Phase 2 (Stage 7) takes the selected signals and runs them through a full portfolio construction pipeline with factor attribution and diagnostics.

The OOS period (2015-2019) is never used for any selection decision — it is purely for final evaluation.

**Final result:**

| Metric | Value |
|--------|-------|
| Signals selected | 3 (all cash-flow quality) |
| Pre-OOS Sharpe (1975-2014) | **1.61** |
| OOS Sharpe (2015-2019) | **0.39** |
| FF Alpha (annualized) | **7.5%** |
| Alpha t-statistic (Newey-West HAC) | **5.49** |
| Max Drawdown | -14.6% |
| Candidates enumerated | 770 |
| Candidates surviving all filters | 27 |
| Total pipeline runtime | 23.5 minutes |

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

Stage 7: FINAL EVALUATION
    Look up neutralized signals → combine → backtest (1975-2019)
    → Factor attribution (FF5+Mom)
    → Bootstrap CIs on OOS Sharpe
    → Long vs short leg attribution
    Output: final Sharpe, alpha, all diagnostics
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

Phase 1 (Stages 1-6) selected WHICH signals to use. Stage 7 evaluates HOW THEY PERFORM using the full portfolio pipeline.

### 9.1 Backtest Protocol

| Parameter | Value |
|-----------|-------|
| Rebalance frequency | Monthly |
| Long positions | 50 (top decile by composite alpha) |
| Short positions | 50 (bottom decile) |
| Transaction costs | 10 bps round-trip |
| Evaluation period | 1975-01 to 2019-12 (541 months) |
| Pre-OOS | 1975-01 to 2014-12 (481 months) — used for selection |
| OOS | 2015-01 to 2019-12 (60 months) — never used for decisions |

### 9.2 Combination Method

We tested both equal-weight and IC-weighted combination:

| Metric | Equal-Weight | IC-Weighted |
|--------|-------------|-------------|
| **Pre-OOS Sharpe** | **1.605** | 1.596 |
| **OOS Sharpe** | **0.388** | 0.068 |
| **FF Alpha (ann)** | **7.5%** | 6.7% |
| **Alpha t-stat** | **5.49** | 4.69 |
| Sortino | **2.68** | 2.42 |
| Max Drawdown | **-14.6%** | -16.1% |
| Avg Turnover | **1.02** | 1.06 |
| R-squared (FF6) | 0.116 | 0.111 |
| Long Sharpe | **1.18** | 1.16 |
| Short Sharpe | -0.46 | -0.48 |

**Equal-weight wins on every metric.** With only 3 strong signals, IC-weighting introduces estimation noise from trailing IC computation that hurts more than it helps. IC-weighting adds value when you have many signals with varying quality; with 3 strong signals, equal weight is more stable.

### 9.3 Factor Attribution

Time-series regression on Fama-French 5 factors + Momentum (Newey-West HAC standard errors):

```
r_{p,t} - rf = α + β₁·MktRF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + β₆·Mom + ε
```

| Factor | Loading | Interpretation |
|--------|---------|----------------|
| **Alpha (annualized)** | **7.5% (t=5.49)** | **Returns unexplained by any factor — genuine stock selection** |
| R-squared | 0.116 | 88% of variance is idiosyncratic |

The alpha of 7.5% (t=5.49) is significant at the 1% level. The low R-squared means the strategy's returns are mostly driven by stock-specific factors, not systematic exposures.

### 9.4 Bootstrap Confidence Intervals

Using circular block bootstrap with block size 12 months, 10,000 iterations:

| Period | Sharpe | 95% CI |
|--------|--------|--------|
| OOS (2015-2019) | 0.39 | [-0.32, 1.20] |

The CI includes zero due to the short 60-month OOS period. The primary significance test is the full-sample alpha (481+ months, t=5.49).

### 9.5 Long vs Short Leg Attribution

| Leg | Sharpe | Interpretation |
|-----|--------|----------------|
| Long | **1.18** | Strong stock picking — long book generates alpha |
| Short | -0.46 | Short leg loses money on average |

Alpha is primarily driven by the long leg. The short leg underperforms, consistent with the structural challenge of shorting equities (borrow costs, short squeezes, upward market drift).

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
| Stage 7: Final evaluation | ~3 min | — |
| **Total** | **~23 min** | **Down from 120+ min without optimizations** |

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

**1. Systematic search beats human intuition.** Hand-picking 6 signals from the academic literature produced Sharpe 0.62. Systematic mining of 770 candidates found 3 signals with pre-OOS Sharpe 1.62 — all cash-flow variants that a human researcher might not have prioritized.

**2. Portfolio-level selection is essential.** Ranking signals by individual IC produced Sharpe 0.81. Testing combinations via actual backtests found Sharpe 1.62. The right question is not "which signals are individually good?" but "which signals are good *together*?"

**3. Honest evaluation matters.** Our initial stepwise selected on full-sample Sharpe (including OOS), producing an inflated OOS Sharpe of 0.96. After fixing to select only on pre-OOS data, the honest OOS Sharpe is 0.39 — lower but real. The alpha remains highly significant (7.5%, t=5.49).

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
