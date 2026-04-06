# Equity Alpha Pipeline — Project Context

## What This Project Is
Factor-neutral long/short U.S. equity alpha research pipeline with systematic signal mining.

## Complete Workflow

The system has TWO phases. Phase 1 decides WHICH signals. Phase 2 decides HOW TO TRADE them.

### Phase 1: Signal Discovery (Stages 1-6)
Replaces human signal selection with systematic mining + portfolio-level testing.

```
Stage 1: ENUMERATE — 770 candidate signals from transformation library
    Transform types: level, ratio, growth, acceleration, difference,
    difference_ratio, negate, momentum, volatility, two-field ratio, analyst
    Applied to: 27 fundamental + 6 price + 8 analyst fields

Stage 2: COMPUTE — pull raw fields, apply math, cache
    Pivot cache: each field pivoted once, reused across candidates
    Output: 743 valid signals (some fail due to missing data)

Stage 3: PRE-SELECT — cascaded quality filters
    3a. Standardize (winsorize 1/99 pct + z-score) — all 743
    3b. GPU batch IC evaluation (RTX 4080, CuPy) — all 743, ~1 min
    3c. IC filter (|ICIR| > 0.15) — 743 → ~138
    3d. Turnover + decile spread (CPU, survivors only) — 138 only, ~5 min
    3e. Full filter (HR>52%, TO<0.60, spread t>2.0) — 138 → ~77
    WHY cascaded: computing turnover for all 743 takes 24 min. For 138 takes 5 min.

Stage 4: VALIDATE + DEDUP
    4a. Holdout validation (2005-2014, separate from dev 1975-2004) — 77 → ~57
    4b. Correlation dedup (< 0.70) — 57 → ~27
    WHY holdout: 770 candidates tested, some pass dev by chance

Stage 5: PRECOMPUTE NEUTRALIZATION
    Neutralize ALL ~54 candidates (27 mined + originals) once upfront
    Uses projection matrix cache: M_t = I - X(X'X)⁻¹X'
    WHY precompute: neutralization is deterministic. Stepwise would
    recompute the same signals hundreds of times without this.

Stage 6: FORWARD STEPWISE PORTFOLIO SELECTION
    Start empty. Each step: test adding each remaining candidate.
    Per trial: look up precomputed neutral signal → combine → backtest → Sharpe
    Selection criterion: pre-OOS Sharpe (1975-2014). OOS (2015-2019) NEVER used for selection.
    Equal-weight combination during selection (zero overfitting risk).
    Stop when no addition improves Sharpe by > 0.01.
    Output: optimal signal set (typically 4-6 signals)
```

### Phase 2: Portfolio Construction (Step 9)
Takes the optimal signals from Phase 1, runs them through the FULL portfolio pipeline.

```
Step 9: FINAL EVALUATION
    Input: 6 optimal signal names from stepwise
    → Look up precomputed neutralized signals
    → IC-weighted combination (adaptive signal weights from trailing IC)
    → Constrained optimizer (dollar/beta/sector neutral, turnover penalty) [TODO]
    → Walk-forward backtest (541 months, 10 bps costs)
    → Factor attribution (FF5+Mom, Newey-West HAC)
    → Bootstrap CIs on OOS Sharpe
    → Long vs short leg attribution
```

### Why Two Phases
- Phase 1 uses EQUAL-WEIGHT + NAIVE L/S to select signals. This has zero tunable
  parameters, preventing overfitting during selection.
- Phase 2 uses IC-WEIGHTED + CONSTRAINED OPTIMIZER for the final portfolio. This
  squeezes more performance from the selected signals, but is only applied once
  to the final set, not during the search.

### Critical Anti-Leakage Rules
- Signal discovery (IC evaluation): development period ONLY (1975-2004)
- Signal validation: holdout period ONLY (2005-2014)
- Stepwise selection criterion: pre-OOS Sharpe (1975-2014) ONLY
- OOS (2015-2019): NEVER used for any decision, only for final reporting
- IC computation: signal at date t, returns at t+1 (next period)
- Quarterly fundamentals: 3-month forward-fill (no filing date look-ahead)

## Tech Stack
- Python 3.12, uv package manager, hatchling build
- GPU: CuPy (cupy-cuda13x) on RTX 4080 SUPER (16GB VRAM)
- CUDA toolkit 13.1 at /opt/cuda/
- Optimization: cvxpy with SCS solver
- Data: CRSP/Compustat (symlinked, 569K rows, 1962-2020)

## How to Run
```bash
uv sync
uv pip install cupy-cuda13x                      # GPU (requires CUDA toolkit)
uv run python scripts/build_sector_mapping.py     # one-time: SEC EDGAR sectors
uv run python scripts/stage_1_features.py         # cache control variables
uv run python scripts/run_full_pipeline.py        # FULL WORKFLOW: mine → stepwise → final eval
```

Individual stages (for debugging / analysis):
```bash
uv run python scripts/stage_2_signals.py          # neutralize + combine + report cards
uv run python scripts/stage_3_backtest.py         # backtest (naive + optimizer)
uv run python scripts/stage_4_ml.py               # ML extension comparison
uv run python scripts/stage_5_robustness.py       # robustness battery
uv run python scripts/stage_7_stepwise.py         # stepwise selection standalone
uv run python scripts/run_mining.py               # mining standalone
```

## Project Structure
```
src/
  config.py                    # PipelineConfig dataclasses
  data/                        # DataPanel, sectors, French factors, cleaner
  factors/                     # Hand-picked signals (momentum, fundamental, risk)
    mined/                     # Auto-generated signals from mining machine
  signals/                     # Registry, z-score, neutralize, combine, report card
  mining/                      # Signal mining machine
    enumeration.py             # Generate candidate specs (770)
    transforms.py              # Transform library (16 types)
    compute.py                 # Compute candidates with pivot cache
    evaluate.py                # IC evaluation (GPU batch + CPU turnover/spread)
    filter.py                  # Quality threshold filtering
    deduplicate.py             # Correlation-based dedup
    stepwise.py                # Forward stepwise portfolio selection
    codegen.py                 # Generate Factor .py files for survivors
    runner.py                  # Mining orchestrator
    config.py                  # Mining thresholds and field classifications
  gpu/                         # GPU acceleration
    backend.py                 # CuPy/numpy abstraction (fallback if no GPU)
    ic_batch.py                # Batched IC computation on GPU
    neutralize_batch.py        # Projection matrix cache for fast neutralization
  ml/                          # ML extension (Ridge, ElasticNet, XGBoost)
    purged_cv.py               # Purged k-fold CV with embargo
    models.py                  # AlphaModel wrappers + ml_combine()
  portfolio/                   # Portfolio construction
    covariance.py              # Ledoit-Wolf shrinkage
    risk_model.py              # Factor model covariance (B @ F @ B' + D)
    construction.py            # Alpha → target weights
    optimization.py            # Constrained L/S optimizer (cvxpy)
    backtest.py                # Walk-forward backtest engine
  analytics/                   # Evaluation
    ic.py                      # IC computation (CPU version)
    performance.py             # Sharpe, Sortino, Calmar, max DD
    risk.py                    # VaR, CVaR, Cornish-Fisher
    attribution.py             # FF factor attribution (Newey-West HAC)
    bootstrap.py               # Block bootstrap for CIs
    statistical_tests.py       # DM test, JK Sharpe equality
  utils/
    logger.py                  # PipelineLogger (file logging + desktop notifications)
```

## Known Issues
- Sector mapping 37.5% coverage (historical/delisted tickers → "Other")
- GPU IC uses date intersection across signals → slightly fewer months than CPU
- Constrained optimizer not yet integrated into Step 9 final evaluation
- OOS period only 60 months → wide bootstrap CIs
