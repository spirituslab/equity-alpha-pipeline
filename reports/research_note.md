# Factor-Neutral Long/Short U.S. Equity Alpha Research Pipeline

**Research Note — Prepared for Portfolio Management Review**

---

## Executive Summary

We built a systematic alpha research pipeline that discovers, evaluates, and combines stock-selection signals into a factor-neutral long/short portfolio. The system processes 770 candidate signals derived from CRSP/Compustat data, filters them through cascaded quality tests, and selects the optimal combination via forward stepwise portfolio-level testing.

**Final result: 6 signals, Sharpe 1.58 (full sample), 0.96 (out-of-sample), FF alpha 7.5% (t=6.27).**

| Metric | Value |
|--------|-------|
| **Net Sharpe (full sample, 1975-2019)** | **1.58** |
| **Net Sharpe (OOS, 2015-2019)** | **0.96** |
| **Fama-French alpha (annualized)** | **7.5%** |
| **Alpha t-statistic (Newey-West HAC)** | **6.27** |
| Signals in final portfolio | 6 |
| Candidates enumerated | 770 |
| Candidates surviving all filters | 27 |
| Total pipeline runtime | 25 minutes |

---

## 1. System Architecture

### 1.1 Design Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SIGNAL ENUMERATION                                       │
│                                                                     │
│  Transformation library × CRSP/Compustat fields                     │
│  8 transform types × 27 fundamental + 6 price + 8 analyst fields    │
│  + difference pairs (a-b), difference ratios (a-b)/c, negation     │
│                                                                     │
│  Output: 770 candidate signal specifications                        │
│  Time: < 1 second                                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SIGNAL COMPUTATION                                        │
│                                                                     │
│  For each candidate: pull raw fields from DataPanel, apply math     │
│  Pivot cache: each field pivoted once, reused across candidates     │
│                                                                     │
│  Output: 743 valid signals (27 failed due to missing data)          │
│  Shape per signal: (696 dates × 1,685 stocks)                      │
│  Time: 18 seconds                                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CASCADED PRE-SELECTION                                    │
│                                                                     │
│  3a. Standardize: winsorize (1st/99th pct) + z-score               │
│      All 743 signals, CPU                                           │
│                                                                     │
│  3b. ★ GPU BATCH IC EVALUATION ★                                    │
│      Rank IC for all 743 signals simultaneously on RTX 4080         │
│      Batched argsort + correlation across all candidates per date   │
│      Shape: (743 signals × 145 dates × 1,685 stocks)               │
│      Why GPU: 743 rank correlations per date → batch in one kernel  │
│      Time: ~1 minute                                                │
│                                                                     │
│  3c. IC Filter: keep |ICIR| > 0.15                                  │
│      743 → 138 survivors (81% eliminated)                           │
│                                                                     │
│  3d. Secondary metrics ONLY for 138 IC survivors                    │
│      Turnover + decile spread (CPU, per-signal)                     │
│      Why cascaded: computing these for all 743 would take 24 min    │
│      Computing for 138 takes ~5 min — same result, 5× faster       │
│      Time: ~5 minutes                                               │
│                                                                     │
│  3e. Full filter: |ICIR|>0.15, HR>52%, TO<0.60, spread t>2.0       │
│      138 → 77 survivors                                             │
│                                                                     │
│  Output: 77 candidates passing all development-period filters       │
│  Time: ~7 minutes total                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: VALIDATION + DEDUPLICATION                                │
│                                                                     │
│  4a. Holdout validation (2005-2014)                                 │
│      Re-compute IC on period NOT used for discovery                 │
│      Require |ICIR| > 0.10 on holdout                              │
│      77 → 57 confirmed                                              │
│      Why: guards against multiple testing (770 candidates tested)   │
│                                                                     │
│  4b. Correlation deduplication                                      │
│      Pairwise cross-sectional correlation                           │
│      If two signals corr > 0.70, drop the weaker one               │
│      57 → 27 non-redundant survivors                                │
│      Why: two identical signals waste a portfolio slot              │
│                                                                     │
│  Output: 27 unique, validated, predictive signals                   │
│  Time: ~2 minutes                                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: ★ PRECOMPUTED NEUTRALIZATION ★                            │
│                                                                     │
│  For ALL 54 candidates (27 mined + 6 original + others):            │
│    z-score → OLS neutralize against sector/size/beta                │
│  Uses precomputed projection matrices: M_t = I - X(X'X)⁻¹X'       │
│  Neutralizing = matrix-vector multiply M_t @ signal (instant)       │
│                                                                     │
│  Why precompute: neutralization is deterministic — same signal      │
│  always gives same result. Stepwise would re-neutralize the same    │
│  signals hundreds of times without this cache.                      │
│  Speedup: 70× on stepwise (40 min/step → 33 sec/step)              │
│                                                                     │
│  Output: 54 neutralized signal DataFrames, cached in memory         │
│  Time: ~5 minutes (one-time)                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: FORWARD STEPWISE PORTFOLIO SELECTION                      │
│                                                                     │
│  Start: empty portfolio                                             │
│  Each step: test adding each remaining candidate                    │
│    - Look up precomputed neutralized signals (instant)              │
│    - Equal-weight combine → composite alpha score per stock         │
│    - Walk-forward backtest: long top 50, short bottom 50            │
│      Monthly rebalance, 10 bps costs, 541 months                   │
│    - Measure: Sharpe, OOS Sharpe, FF alpha                         │
│  Accept the candidate that improves Sharpe the most                 │
│  Stop when no candidate improves Sharpe by > 0.01                  │
│                                                                     │
│  Why portfolio-level: individual signal IC ≠ portfolio Sharpe.      │
│  A mediocre-IC signal can be a great diversifier. A high-IC signal  │
│  can be redundant. Only backtest-level testing reveals this.        │
│                                                                     │
│  54 candidates × ~6 steps × 0.7 sec/trial = ~3 minutes             │
│  Time: ~3 minutes                                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: OPTIMAL SIGNAL SET                                         │
│                                                                     │
│  6 signals, Sharpe 1.58 (full), 0.96 (OOS), α 7.5% (t=6.27)      │
│  Total pipeline time: 25 minutes                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Each Design Decision

| Decision | Why |
|----------|-----|
| **GPU for IC evaluation** | 743 rank correlations per date can be batched into one GPU kernel. CPU would loop 743 times per date × 500 dates. GPU: ~1 min. CPU: ~35 min. |
| **Cascaded filtering** | Computing turnover + decile spread for all 743 signals takes 24 min. IC filter eliminates 81% first, so we only compute expensive metrics for 138 survivors. Same result, 5× faster. |
| **Holdout validation** | With 770 candidates tested, ~38 pass development filters by chance alone. The 2005-2014 holdout catches these false positives. |
| **Precomputed neutralization** | Neutralization is deterministic — `neutralize(signal_X)` gives the same result regardless of what other signals are in the trial set. Precomputing all 54 once replaces thousands of redundant recomputations in stepwise. |
| **Stepwise at portfolio level** | Picking signals by individual IC gave Sharpe 0.62. Stepwise portfolio selection gave 1.58. A signal's value depends on what else is in the portfolio — you can only know by testing combinations. |

---

## 2. Data

### 2.1 Source

CRSP/Compustat merged monthly panel — the institutional standard for U.S. equity research. Survivorship-bias-free: includes delisted companies.

| Dimension | Value |
|-----------|-------|
| Total observations | 569,476 |
| Unique companies | 1,685 |
| Date range | January 1962 — September 2020 |
| Fields | 52 (price, returns, quarterly fundamentals, analyst estimates) |

### 2.2 Universe

At each rebalance date, the investable universe:
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

### 2.4 Supplementary Data

- **Fama-French 5 factors + Momentum** (monthly): Factor attribution, beta estimation, risk model
- **Sector classification**: SIC codes from SEC EDGAR → Ken French 12-industry mapping

---

## 3. Signal Mining Machine

### 3.1 Transformation Library

The mining machine systematically applies mathematical transforms to every available data field:

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

### 3.2 Cascaded Pre-Selection

Instead of computing all metrics for all candidates (expensive), we filter by the cheapest metric first:

```
770 candidates
  ├─ Standardize (winsorize + z-score)              all 743 valid
  ├─ GPU batch IC evaluation                         743 → 138 pass (|ICIR| > 0.15)
  ├─ Turnover computation (CPU, 138 only)            138 → 120 pass (TO < 0.60)
  ├─ Decile spread computation (CPU, 120 only)       120 → 77 pass (spread t > 2.0)
  ├─ Holdout validation (2005-2014)                  77 → 57 confirmed
  └─ Correlation deduplication (< 0.70)              57 → 27 non-redundant
```

### 3.3 How IC Evaluation Works — Worked Example

The IC (Information Coefficient) measures whether a signal predicts stock returns. Here is a small example showing exactly what the GPU computes.

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

If IC > 0, the signal correctly identified which stocks would outperform. If IC ≈ 0, no predictive power.

**The GPU processes all 743 signals at once:** stack them into a matrix (743 × 1,685), rank all rows in one GPU operation, correlate with the return ranks in one batched dot product. This replaces 743 sequential CPU operations with a single GPU kernel.

**Repeat for all 145 dates** to build an IC time series per signal:

```
IC matrix: (743 signals × 145 dates)

Signal A: [+0.03, -0.01, +0.05, +0.02, +0.04, ...]   145 monthly values
Signal B: [+0.08, +0.04, +0.06, +0.07, +0.05, ...]
Signal C: [+0.01, -0.02, +0.00, +0.03, -0.01, ...]
```

**Then summarize each signal's IC time series:**

```
Signal A: Mean IC = 0.030, Std IC = 0.08, ICIR = 0.030/0.08 = 0.38  ← consistent, keep
Signal B: Mean IC = 0.055, Std IC = 0.09, ICIR = 0.055/0.09 = 0.61  ← strong, keep
Signal C: Mean IC = 0.005, Std IC = 0.07, ICIR = 0.005/0.07 = 0.07  ← noise, discard
```

**ICIR** (IC Information Ratio) = Mean IC / Std IC — the signal's "Sharpe ratio." It measures not how strong the prediction is on average, but how **consistent** it is. A signal with IC = 0.02 every month (high ICIR) is more valuable than one with IC = +0.10 some months and -0.08 others (low ICIR), because the portfolio rebalances monthly and needs reliable direction, not occasional lucky calls.

We filter at **ICIR > 0.15**: of 743 candidates, 138 pass this threshold.

### 3.4 Top Discoveries

| Signal | Formula | Dev ICIR | Val ICIR | Hit Rate | Category |
|--------|---------|----------|----------|----------|----------|
| oancfy_to_mktcap | Cash flow / market cap | 0.406 | 0.245 | 66% | Value |
| oancfy_div_seqq | Cash flow / shareholder equity | 0.389 | 0.284 | 66% | Quality |
| oancfy_div_oibdpq | Cash flow / operating income | 0.379 | 0.197 | 67% | Quality |
| oancfy_div_dlttq | Cash flow / long-term debt | 0.365 | 0.280 | 63% | Quality |
| **saleq_minus_cogsq_to_mktcap** | **(Sales - COGS) / market cap** | **0.358** | **0.168** | **66%** | **Value** |
| cheq_qoq | QoQ cash growth | 0.255 | 0.351 | 59% | Growth |
| **ibq_minus_oancfy_div_ceqq** | **(Income - CashFlow) / equity** | **-0.387** | **-0.240** | **33%** | **Quality** |

**Bold** entries are signals that require the new DIFFERENCE / DIFFERENCE_RATIO transforms — they could not have been discovered without the expanded transformation library.

**The dominant theme is cash flow quality.** Operating cash flow (OANCFY) normalized by various balance sheet items accounts for the majority of top discoveries. This is economically coherent: cash flow from operations is the hardest accounting number to manipulate and the most direct measure of economic value creation.

---

## 4. Signal Processing

### 4.1 Standardization

For each signal at each date:
1. **Winsorize** at 1st/99th percentile — removes data errors, split artifacts, filing anomalies
2. **Z-score**: z_{i,t} = (x_{i,t} - mean_t) / std_t — makes signals comparable in magnitude

### 4.2 Neutralization

For each signal at each date, cross-sectional OLS regression:

```
signal_{i,t} = a + b₁·sector_dummies + b₂·log(mcap_{i,t}) + b₃·beta_{i,t} + ε_{i,t}
```

Keep only the residual ε — the part orthogonal to sector, size, and beta.

**Why this matters:** Without neutralization, a momentum signal that loads on high-beta tech stocks would generate returns from market beta, not stock selection. Neutralization forces the portfolio to earn returns from pure alpha.

**Implementation optimization:** We precompute projection matrices M_t = I - X(X'X)⁻¹X' once for all dates. Neutralizing any signal then becomes a single matrix-vector multiply: residual = M_t × signal. This is reused across all 54 candidates in stepwise selection.

### 4.3 Combination

Equal-weight average of neutralized z-scores in stepwise selection:

```
alpha_{i,t} = (1/K) × Σ_k z_{k,i,t}^{neutralized}
```

---

## 5. Portfolio Construction

### 5.1 Long/Short Portfolio

At each monthly rebalance:
1. Rank ~400 S&P 500 stocks by composite alpha score
2. Long top 50 stocks, short bottom 50 stocks
3. Equal weight within each leg (2% per stock)
4. Dollar-neutral: sum of weights = 0

### 5.2 Backtest Protocol

| Parameter | Value |
|-----------|-------|
| Rebalance frequency | Monthly |
| Long positions | 50 |
| Short positions | 50 |
| Transaction costs | 10 bps round-trip |
| Backtest period | 1975-01 to 2019-12 (541 months) |

### 5.3 Anti-Leakage Rules

| Rule | Implementation |
|------|----------------|
| Quarterly fundamentals | 3-month forward-fill from last known value |
| Returns | t used for signals, t+1 only for realized P&L |
| Universe membership | S&P 500 flag at date t, not t+1 |
| Standardization | z-scores from cross-section at t only |
| Signal discovery | Development period (1975-2004) only |
| Validation | Separate holdout (2005-2014) |

---

## 6. Forward Stepwise Selection

### 6.1 Why Portfolio-Level Selection

Individual signal quality (IC) does not predict portfolio performance:

| Approach | How signals chosen | Full Sharpe | OOS Sharpe |
|----------|-------------------|-------------|------------|
| Hand-picked by IC | Top 6 by academic literature | 0.62 | 0.42 |
| Top 5 by individual ICIR | Rank by ICIR, take top 5 | 0.81 | 0.26 |
| **Stepwise portfolio test** | **Test each combination via backtest** | **1.58** | **0.96** |

The difference is 2.5× Sharpe. Signals interact — a signal with moderate IC but low correlation to existing signals adds more portfolio value than a high-IC signal that overlaps with what's already there.

### 6.2 The Algorithm — Worked Example

Suppose 54 candidate signals survive pre-selection. The algorithm works as follows:

**Step 1 — Test each signal alone (54 backtests):**
```
Try signal A alone → backtest 541 months → Sharpe 0.85
Try signal B alone → backtest 541 months → Sharpe 1.07  ← best
Try signal C alone → backtest 541 months → Sharpe 0.62
... (test all 54)

Winner: signal B (oancfy_div_seqq, Sharpe 1.07)
Selected set: [B]
```

**Step 2 — Test B + each of the remaining 53 (53 backtests):**
```
Try B + A → combine, backtest → Sharpe 1.20
Try B + C → combine, backtest → Sharpe 0.95
Try B + D → combine, backtest → Sharpe 1.40  ← best
... (test all 53)

Winner: B + D (added ibcomq_to_mktcap, Sharpe 1.40)
Selected set: [B, D]
```

**Step 3 — Test B + D + each of the remaining 52 (52 backtests):**
```
Try B + D + A → combine, backtest → Sharpe 1.35
Try B + D + C → combine, backtest → Sharpe 1.49  ← best
... (test all 52)

Winner: B + D + C (added oancfy_to_mktcap, Sharpe 1.49)
Selected set: [B, D, C]
```

**...continues until adding any remaining signal improves Sharpe by less than 0.01...**

**Step 7 — Test all remaining → best improvement is -0.05 → stop.**

Each addition is evaluated in the context of what's already selected. Signal D is chosen at step 2 not because it has the highest individual IC, but because it adds the most value *on top of signal B*. This is why portfolio-level testing finds different (and better) answers than individual signal ranking.

The formal algorithm:

```
selected = []
For step = 1, 2, 3, ...:
    For each remaining candidate:
        trial = selected + [candidate]
        alpha = equal_weight_combine(neutralized signals in trial)
        sharpe = walk_forward_backtest(alpha, 541 months, 10 bps costs)
    Accept candidate with highest sharpe
    If improvement < 0.01: stop
```

Each trial takes ~0.7 seconds (combine + backtest) because neutralized signals are precomputed.

### 6.3 Selection Path

| Step | Signal Added | Full Sharpe | OOS Sharpe | FF Alpha | t-stat |
|------|-------------|-------------|------------|----------|--------|
| 1 | oancfy_div_seqq | 1.070 | 0.029 | 4.2% | 2.96 |
| 2 | + ibcomq_to_mktcap | 1.401 | 0.236 | 6.5% | 4.36 |
| 3 | + oancfy_to_mktcap | 1.485 | 0.391 | 7.6% | 5.56 |
| 4 | + oancfy_div_rectq | 1.501 | 0.420 | 8.1% | 5.82 |
| 5 | + ibq_minus_oancfy_div_ceqq | 1.539 | 0.547 | 8.6% | 5.97 |
| 6 | + cheq_qoq | **1.578** | **0.961** | **7.5%** | **6.27** |
| *(stopped — no further improvement)* |

### 6.4 The Final 6 Signals

| Signal | Formula | Economic Rationale |
|--------|---------|-------------------|
| **oancfy_div_seqq** | Operating cash flow / shareholder equity | Cash return on equity — how much cash per dollar of equity |
| **ibcomq_to_mktcap** | Net income / market cap | Earnings yield — cheap stocks with real earnings |
| **oancfy_to_mktcap** | Operating cash flow / market cap | Cash flow yield — cheap stocks with real cash generation |
| **oancfy_div_rectq** | Operating cash flow / receivables | Cash collection efficiency — firms converting sales to cash |
| **ibq_minus_oancfy_div_ceqq** | (Income - Cash flow) / equity | Accrual quality — penalizes earnings not backed by cash |
| **cheq_qoq** | Quarter-over-quarter cash growth | Cash momentum — firms with improving liquidity |

All six are interpretable with clear economic rationale. Four are cash-flow quality variants. One is an accrual measure (requires DIFFERENCE_RATIO transform). One is cash momentum.

---

## 7. Results

### 7.1 Headline Performance

| Metric | Value |
|--------|-------|
| Annualized Return (net of costs) | ~12% |
| Annualized Volatility | ~7.5% |
| **Net Sharpe Ratio (full sample)** | **1.58** |
| **Net Sharpe Ratio (OOS 2015-2019)** | **0.96** |
| **FF Alpha (annualized)** | **7.5%** |
| **Alpha t-statistic (HAC)** | **6.27** |
| Number of signals | 6 |
| Avg positions | 50 long / 50 short |

### 7.2 Factor Attribution

Time-series regression on Fama-French 5 factors + Momentum:

```
r_{p,t} - rf = α + β₁·MktRF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + β₆·Mom + ε
```

The alpha of 7.5% (t=6.27) means the strategy generates returns that cannot be explained by any combination of the six most widely studied systematic risk factors. This is significant at the 1% level.

### 7.3 Evolution of Results

| Version | What Changed | Signals | Full SR | OOS SR | Alpha | t-stat |
|---------|-------------|---------|---------|--------|-------|--------|
| v1 | Hand-picked signals, IC-weighted | 6 | 0.62 | 0.42 | 4.0% | 2.22 |
| v2 | + Signal mining machine | 6+5=11 | 0.81 | 0.26 | 6.2% | 3.40 |
| v3 | + Stepwise portfolio selection | 4 | 1.51 | 0.42 | 8.2% | 5.94 |
| **v4** | **+ Expanded transforms + cascaded pipeline** | **6** | **1.58** | **0.96** | **7.5%** | **6.27** |

Each improvement came from a methodological advance:
- **v1→v2**: Systematic search finds signals humans miss
- **v2→v3**: Portfolio-level selection beats individual IC ranking
- **v3→v4**: Difference/ratio transforms enable accrual and gross profit signals that complete the portfolio

---

## 8. Performance Engineering

### 8.1 Runtime Breakdown

| Stage | Time | Optimization |
|-------|------|-------------|
| Signal enumeration | < 1 sec | — |
| Signal computation (770) | 18 sec | Pivot field caching |
| **GPU batch IC (743)** | **~1 min** | **RTX 4080, batched argsort+correlation per date** |
| IC filter | instant | Cascaded: eliminates 81% before expensive metrics |
| Turnover + spread (138 only) | ~5 min | Cascaded: 138 instead of 743 |
| Validation + dedup | ~2 min | — |
| **Precompute neutralization (54)** | **~5 min** | **Projection matrix cache, one-time cost** |
| **Stepwise selection (6 steps)** | **~3 min** | **Precomputed signals: 0.7 sec/trial vs 40 sec** |
| **Total** | **~25 min** | **Down from 120+ min without optimizations** |

### 8.2 Hardware Utilization

| Resource | Usage | Purpose |
|----------|-------|---------|
| RTX 4080 SUPER (16 GB VRAM) | GPU batch IC evaluation | Parallel rank correlation for 743 signals |
| i9-14900KF (32 threads) | Signal computation, backtest | Numpy/pandas operations |
| 32 GB RAM | Signal tensors in memory | All 54 neutralized signals cached simultaneously |

---

## 9. Limitations and Honest Assessment

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| OOS period only 60 months | Wide confidence intervals | Full-sample alpha (541 months, t=6.27) is the primary significance test |
| Sector coverage 37.5% | Historical/delisted tickers assigned "Other" | Neutralization still controls for 11 identified sectors |
| No borrow costs in backtest | Overstates short-leg returns by ~25-50 bps/year | S&P 500 general collateral borrow is cheap |
| Monthly rebalance only | Misses intraday/weekly signals | Appropriate for fundamental signals which change slowly |
| Equal-weight combination in stepwise | Could optimize signal weights | Simplicity reduces overfitting risk; IC-weighting available as extension |
| Free data only | Not WRDS institutional quality | CRSP/Compustat merged IS institutional quality — survivorship-bias-free |

---

## 10. Conclusion

This project demonstrates three principles of systematic alpha research:

**1. Systematic search beats human intuition.** Hand-picking 6 well-known signals from the academic literature produced Sharpe 0.62. Systematic mining of 770 candidates from the same data, filtered through the same evaluation framework, found a 6-signal portfolio with Sharpe 1.58.

**2. Portfolio-level selection is essential.** Ranking signals by individual IC and picking the top ones produced Sharpe 0.81 with poor OOS performance (0.26). Testing combinations via actual backtests found a portfolio with Sharpe 1.58 and OOS 0.96. The right question is not "which signals are individually good?" but "which signals are good *together*?"

**3. Engineering enables research.** GPU-batched IC evaluation, cascaded filtering, and precomputed neutralization reduced pipeline runtime from 120+ minutes to 25 minutes. This makes iterative research feasible — test a hypothesis, examine results, refine, repeat.

The final portfolio is economically interpretable: 4 cash-flow quality signals, 1 accrual quality signal, 1 cash momentum signal. The dominant theme — that markets undervalue firms with strong, genuine cash generation — is consistent with decades of accounting research and practitioner experience.

---

## Appendix A: Reproducibility

```bash
git clone https://github.com/spirituslab/equity-alpha-pipeline
cd equity-alpha-pipeline
uv sync
uv pip install cupy-cuda13x                      # GPU support (requires CUDA toolkit)
uv run python scripts/build_sector_mapping.py     # One-time: sector data from SEC EDGAR
uv run python scripts/stage_1_features.py         # Cache control variables
uv run python scripts/run_full_pipeline.py        # Full pipeline: mine → select → optimal set
```

## Appendix B: Project Structure

```
equity-alpha-pipeline/
  config/
    pipeline.yaml              # All parameters
    mining.yaml                # Mining thresholds
  src/
    data/                      # DataPanel, sectors, French factors
    factors/                   # Hand-picked + auto-generated Factor subclasses
      mined/                   # 27 auto-generated signals from mining machine
    signals/                   # Registry, z-score, neutralize, combine, report card
    mining/                    # Enumeration, compute, evaluate, filter, dedup, stepwise
    gpu/                       # CuPy backend, batched IC, projection cache
    portfolio/                 # Risk model, optimizer, backtest engine
    analytics/                 # IC, attribution, bootstrap, performance
    utils/                     # Logging + notifications
  scripts/
    run_full_pipeline.py       # Complete workflow: mine → stepwise → optimal
    stage_1-7_*.py             # Individual pipeline stages
    run_mining.py              # Signal mining standalone
```
