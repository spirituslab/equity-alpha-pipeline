# Nested Chronological Validation — Workflow Diagram

## Pipeline Overview

```
+==============================================================================+
|                    NESTED VALIDATION PIPELINE v2                              |
|                    config/pipeline.yaml -> mode: "nested"                    |
|                                                                              |
|  Key improvements over v1:                                                   |
|    - Method-agnostic: every trial evaluates EW + IC-weighted + inverse-vol   |
|    - Parallel: 16 cores evaluate candidates simultaneously per step          |
|    - Flushed logging: real-time progress via tail -f logs/*.log              |
+==============================================================================+

+------------------------------------------------------------------------------+
|  PHASE 1: GLOBAL PRECOMPUTATION (done once, shared across all folds)         |
|  ~8 min total, GPU-heavy                                                     |
|                                                                              |
|  +==============+    +==============+    +==================+                |
|  | 1. ENUMERATE  |--->| 2. COMPUTE   |--->| 3. STANDARDIZE   |                |
|  | 770 candidate |    | raw signals  |    | winsorize+zscore |                |
|  | specs         |    | (pivot cache)|    | all 743 signals  |                |
|  +==============+    +==============+    +========|=========+                |
|                                                    |                          |
|                                                    v                          |
|  +================================================================+         |
|  | 4. GPU GLOBAL PRECOMPUTE                                        |         |
|  |                                                                  |         |
|  |  +================+  +==================+  +================+   |         |
|  |  | 4a. Projection |  | 4b. GPU IC filter |  | 4c. GPU batch  |   |         |
|  |  | cache: M_t =   |  | on full inner     |  | turnover +     |   |         |
|  |  | I-X(X'X)^-1*X' |  | period (->2009)   |  | decile spread  |   |         |
|  |  | (all dates)    |  | 743 -> ~150       |  | for 150 surv.  |   |         |
|  |  +================+  +==================+  +================+   |         |
|  |                              |                                   |         |
|  |                              v                                   |         |
|  |  +===========================================================+  |         |
|  |  | 4d. NEUTRALIZE all ~150 IC survivors via GPU               |  |         |
|  |  | residual = M_t @ signal  (matrix-vector multiply)          |  |         |
|  |  | -> precomputed_neutral dict (cached in memory, ~18GB)      |  |         |
|  |  +===========================================================+  |         |
|  +================================================================+         |
|                                                                              |
|  5. PER-FOLD GPU IC (sequential on GPU, ~1 min each)                        |
|     Fold 1 IC (->1989) | Fold 2 IC (->1994) | Fold 3 (->1999) | Fold 4     |
+------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------------------------------------------------+
|  PHASE 2: INNER LAYER -- 4 EXPANDING FOLDS (1975-2009)                      |
|  Folds run sequentially. Within each fold, stepwise is parallelized.         |
|                                                                              |
|  +=================+ +=================+ +=================+ +============+  |
|  | FOLD 1          | | FOLD 2          | | FOLD 3          | | FOLD 4     |  |
|  | Train: 1975-89  | | Train: 1975-94  | | Train: 1975-99  | | Train:     |  |
|  | Val:   1990-94  | | Val:   1995-99  | | Val:   2000-04  | | 1975-04    |  |
|  +=================+ +=================+ +=================+ | Val:       |  |
|                                                               | 2005-09    |  |
|  PER-FOLD PIPELINE (same for each):                          +============+  |
|                                                                              |
|  +========+   +==========+   +=========+   +========+                       |
|  | IC     |-->| IC+TO+   |-->| Holdout |-->| Corr.  |                       |
|  | filter |   | Spread   |   | Val on  |   | Dedup  |                       |
|  | ~150   |   | filters  |   | fold's  |   | <0.70  |                       |
|  | ->~80  |   |          |   | val per.|   |        |                       |
|  +========+   +==========+   +=========+   +===|====+                       |
|                                                  |                           |
|                                                  v                           |
|  +============================================================+             |
|  | FORWARD STEPWISE (method-agnostic, parallel)                |             |
|  |                                                             |             |
|  | Step 1: test 22 candidates                                  |             |
|  |   +------+------+------+------+------+------+              |             |
|  |   | Cand | Cand | Cand | Cand | Cand | ...  |  16 cores    |             |
|  |   |  A   |  B   |  C   |  D   |  E   |      |  in parallel |             |
|  |   +--+---+--+---+--+---+--+---+--+---+------+              |             |
|  |      |      |      |      |      |                          |             |
|  |      v      v      v      v      v                          |             |
|  |   Each candidate evaluated under 3 methods:                 |             |
|  |   +----------+  +--------------+  +-------------+          |             |
|  |   | EW       |  | IC-weighted  |  | Inverse-vol |          |             |
|  |   | combine  |  | combine      |  | combine     |          |             |
|  |   | backtest |  | backtest     |  | backtest    |          |             |
|  |   | SR=0.72  |  | SR=0.68      |  | SR=0.70     |          |             |
|  |   +----------+  +--------------+  +-------------+          |             |
|  |         \              |               /                    |             |
|  |          +-----> mean(0.72, 0.68, 0.70) = 0.70 <----+     |             |
|  |                  METHOD-AGNOSTIC SCORE                      |             |
|  |                                                             |             |
|  | Step 2: test 21 remaining (best from step 1 removed)        |             |
|  |   ... same parallel + 3-method pattern ...                  |             |
|  |                                                             |             |
|  | Stop when mean/median Sharpe stops improving                |             |
|  | -> 4-6 selected signals per fold                            |             |
|  +============================================================+             |
|                                                                              |
|  Output per fold:                                                            |
|    FoldResult { stepwise_selected, per_signal_metrics, fold_sharpe }         |
|    Saved to outputs/runs/{timestamp}/inner_folds/fold_N/                     |
+------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------------------------------------------------+
|  PHASE 3: CROSS-FOLD STABILITY                                              |
|                                                                              |
|  +===================================================================+      |
|  |  StabilityTracker                                                  |      |
|  |                                                                    |      |
|  |  For each signal, count: how many folds did it survive?            |      |
|  |                                                                    |      |
|  |  Signal A:  Fold1 Y  Fold2 Y  Fold3 Y  Fold4 Y  -> stab=1.00    |      |
|  |  Signal B:  Fold1 Y  Fold2 N  Fold3 Y  Fold4 N  -> stab=0.50    |      |
|  |  Signal C:  Fold1 Y  Fold2 N  Fold3 N  Fold4 N  -> stab=0.25    |      |
|  |                                                                    |      |
|  |  Keep: min_fold_survival >= 2 AND stability >= 0.5                |      |
|  |  -> stable_candidates (A, B survive; C dropped)                   |      |
|  +===================================================================+      |
|                                                                              |
|  Saved to outputs/runs/{timestamp}/stability/                               |
+------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------------------------------------------------+
|  PHASE 4: NESTED STEPWISE SELECTION (method-agnostic, parallel)             |
|                                                                              |
|  +===================================================================+      |
|  |  forward_stepwise_nested(stable_candidates)                        |      |
|  |                                                                    |      |
|  |  Step 1: test 15 stable candidates (16 cores in parallel)         |      |
|  |                                                                    |      |
|  |    Each candidate evaluated across ALL 4 folds x 3 methods:       |      |
|  |    +-----------------------------------------------------------+  |      |
|  |    | Candidate D:                                               |  |      |
|  |    |   Fold 1: EW=0.72, IC=0.68, IV=0.70 -> avg=0.70          |  |      |
|  |    |   Fold 2: EW=0.81, IC=0.79, IV=0.80 -> avg=0.80          |  |      |
|  |    |   Fold 3: EW=0.68, IC=0.65, IV=0.67 -> avg=0.67          |  |      |
|  |    |   Fold 4: EW=0.75, IC=0.73, IV=0.74 -> avg=0.74          |  |      |
|  |    |                                                            |  |      |
|  |    |   mean_sharpe = 0.73                                       |  |      |
|  |    |   penalized_score = 0.73                                   |  |      |
|  |    |     - 0.1 x turnover                                      |  |      |
|  |    |     - 0.2 x instability                                   |  |      |
|  |    |     - 0.1 x concentration                                 |  |      |
|  |    +-----------------------------------------------------------+  |      |
|  |                                                                    |      |
|  |  Stop when ANY of:                                                 |      |
|  |    - mean Sharpe improvement < 0                                   |      |
|  |    - median Sharpe improvement < 0                                 |      |
|  |    - any fold's turnover worsens > 0.15                           |      |
|  |    - any fold's max drawdown worsens > 0.10                       |      |
|  |                                                                    |      |
|  |  -> FROZEN signal set (typically 4-6 signals)                      |      |
|  +===================================================================+      |
|                                                                              |
|  freeze_manifest.yaml (signal names, config, git hash)                      |
+------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------------------------------------------------+
|  PHASE 5: MIDDLE LAYER -- MODEL COMPARISON (2010-2014)                      |
|  Frozen signals only. No new signals, no threshold edits.                    |
|  NOW A FAIR RACE: signals were NOT selected to favor any method              |
|                                                                              |
|  +==============+    +==============+    +==============+                    |
|  | Equal-Weight  |    | IC-Weighted   |    | Inverse-Vol  |                    |
|  | combine       |    | combine       |    | combine      |                    |
|  |      |        |    |      |        |    |      |       |                    |
|  | Backtest on   |    | Backtest on   |    | Backtest on  |                    |
|  | 2010-2014     |    | 2010-2014     |    | 2010-2014    |                    |
|  |      |        |    |      |        |    |      |       |                    |
|  | Sharpe: X.XX  |    | Sharpe: X.XX  |    | Sharpe: X.XX |                    |
|  +======|=======+    +======|=======+    +======|=======+                    |
|         +==============+====+====================+                            |
|                        v                                                      |
|  +===================================================================+      |
|  |  Diebold-Mariano pairwise tests                                    |      |
|  |  If no method significantly better than EW -> default to EW        |      |
|  |  -> selected_method                                                |      |
|  +===================================================================+      |
|                                                                              |
|  Saved to outputs/runs/{timestamp}/model_comparison/                        |
+------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------------------------------------------------+
|  PHASE 6: STAGE 7 -- OOS-ONLY EVALUATION (2015-2019)                       |
|  Touched ONCE. No optimization. Pure out-of-sample test.                     |
|                                                                              |
|  +===================================================================+      |
|  |  7a. Combine frozen signals with selected_method                   |      |
|  |  7b. Walk-forward backtest (warm-up ~2005, report only 2015-2019) |      |
|  |  7c. OOS metrics: Sharpe, Sortino, Calmar, MaxDD, Win Rate        |      |
|  |  7d. Factor attribution (FF5+Mom, Newey-West HAC) on OOS only     |      |
|  |  7e. Bootstrap 95% CI on OOS Sharpe + p-value                     |      |
|  |  7f. Deflated Sharpe Ratio (corrects for 770 trials tested)       |      |
|  |      Probability of Backtest Overfitting (CSCV across folds)      |      |
|  |  7g. Subperiod stability (OOS Early 2015-17, Late 2018-19)        |      |
|  |  7h. Cost sensitivity (0-30 bps, OOS only)                        |      |
|  |  7i. Regime analysis (bull/bear, OOS only)                        |      |
|  |  7j. Signal ablation (drop-one-out, OOS only)                     |      |
|  |  7k. Quantile monotonicity (decile returns, OOS only)             |      |
|  +===================================================================+      |
|                                                                              |
|  Saved to outputs/runs/{timestamp}/oos_evaluation/                          |
+------------------------------------------------------------------------------+
```


## Parallelism Model

```
WITHIN EACH STEPWISE STEP:
=========================

  Sequential (must wait for previous step to pick winner):
    Step 1 -> Step 2 -> Step 3 -> Step 4 -> Step 5 -> STOP

  Parallel WITHIN each step (candidates are independent):
    Step 3: test 20 remaining candidates
    
    ProcessPoolExecutor(max_workers=16, fork)
    
    Core  1: [Cand A: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  2: [Cand B: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  3: [Cand C: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  4: [Cand D: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  5: [Cand E: EW backtest | IC backtest | IV backtest] -> avg SR
    ...
    Core 16: [Cand P: EW backtest | IC backtest | IV backtest] -> avg SR
    --- wait ---
    Core  1: [Cand Q: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  2: [Cand R: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  3: [Cand S: EW backtest | IC backtest | IV backtest] -> avg SR
    Core  4: [Cand T: EW backtest | IC backtest | IV backtest] -> avg SR
    --- all done ---
    
    Pick best -> add to portfolio -> next step

    Serial: 20 cand x 3 methods x 1 sec = 60 sec
    Parallel (16 cores): ceil(20/16) x 3 sec = 6 sec   ~10x speedup


ACROSS FOLDS (sequential to avoid nested pool contention):
==========================================================

    Fold 1 stepwise (16-core parallel within)   ~2-3 min
    -> Fold 2 stepwise (16-core parallel within) ~2-3 min
    -> Fold 3 stepwise (16-core parallel within)  ~2-3 min
    -> Fold 4 stepwise (16-core parallel within)   ~2-3 min

    Total fold phase: ~10-12 min (vs ~30 min single-threaded)
```


## Method-Agnostic Selection: Why It Matters

```
OLD (EW-biased):                          NEW (method-agnostic):
================                          =====================

Phase 2 stepwise:                         Phase 2 stepwise:
  score = Sharpe(EW_combine)                score = mean(Sharpe(EW), 
                                                        Sharpe(IC),
                                                        Sharpe(IV))

  Selects signals that are                  Selects signals that are
  equally strong (EW rewards this)          FUNDAMENTALLY PREDICTIVE
       |                                         |
       v                                         v
Phase 5 model comparison:                 Phase 5 model comparison:
  EW always wins (rigged race)              Fair race -- any method can win
  IC-weighted/IV penalized                  because signals weren't chosen
  for signals they didn't choose            to favor any weighting scheme
```


## Data Flow: What Period Each Phase Sees

```
  1962        1975        1990   1995   2000   2005   2010   2015   2020
   |           |           |      |      |      |      |      |      |
   | burn-in   |<=============== INNER LAYER ================>|      |
   | (discard) |           |      |      |      |      |      |      |
   |           |           |      |      |      |      |      |      |
   |           |  Fold 1:  |TRAIN-------------------->|VAL-->|       |
   |           |  Fold 2:  |TRAIN------------------------->|VAL-->| |
   |           |  Fold 3:  |TRAIN------------------------------>|VAL-->|
   |           |  Fold 4:  |TRAIN----------------------------------->|VAL-->|
   |           |           |      |      |      |      |      |      |
   |           |           |      |      |      |      |MIDDLE|      |
   |           |           |      |      |      |      |2010- |      |
   |           |           |      |      |      |      |2014  |      |
   |           |           |      |      |      |      |      |      |
   |           |           |      |      |      |      |      | OOS  |
   |           |           |      |      |      |      |      |2015- |
   |           |           |      |      |      |      |      |2019  |
   +-----------+-----------+------+------+------+------+------+------+
```


## Estimated Runtime (RTX 4080 SUPER, 16-core)

```
  Phase 1: Global precompute                    ~8 min
    Step 1-3: enumerate + compute + standardize   3 min
    Step 4a: projection cache                     0.5 min
    Step 4b: GPU IC filter                        1 min
    Step 4c: GPU turnover + spread                0.5 min
    Step 4d: GPU neutralize 150 signals           2 min
    Step 5: per-fold GPU IC (4x)                  4 min
    
  Phase 2: Inner folds (4 sequential)           ~10-12 min
    Per fold: filter + dedup + parallel stepwise  ~2.5-3 min
    (16 cores x 3 methods per candidate)
    
  Phase 3: Stability tracking                   ~1 sec
  
  Phase 4: Nested stepwise (parallel)           ~3-5 min
    ~15 candidates x ~6 steps x 4 folds x 3 methods
    parallelized across 16 cores
    
  Phase 5: Model comparison                     ~2 min
    3 methods x backtest on 2010-2014
    
  Phase 6: OOS evaluation                       ~3 min
    backtest + bootstrap + DSR/PBO + ablation
    
  TOTAL: ~25-30 min (vs ~48 min single-core EW-only)
```


## Logging (real-time via tail -f)

```
  $ tail -f logs/full_pipeline_*.log

  22:50:17 | >>> Nested Mining
  22:50:17 | Step 1: Generated 770 candidate specifications (1s)
  22:50:38 | Step 2: Computed 743 signals (21s)
  22:50:40 | Step 3: Standardized 743 signals (2s)
  22:51:10 | Step 4a: Cached 632 projection matrices (30s)
  22:52:15 | Step 4b: Loose IC filter: 152/743 passed (65s)
  22:52:45 | Step 4c: Turnover + spread done (30s)
  22:54:45 | Step 4d: All 152 signals neutralized (120s)
  22:55:50 | [Step 5] Fold 1 IC (->1989) done (65s)
  22:56:55 | [Step 5] Fold 2 IC (->1994) done (65s)
  22:58:00 | [Step 5] Fold 3 IC (->1999) done (65s)
  22:59:05 | [Step 5] Fold 4 IC (->2004) done (65s)
  22:59:05 | Step 6: Running 4 inner folds...
  22:59:10 | [Fold 1] IC filter: 82/152 passed
  22:59:10 | [Fold 1] Dev filter: 45/82 passed
  22:59:11 | [Fold 1] Val filter: 31/45 passed
  22:59:11 | [Fold 1] Dedup: 22/31 passed
  22:59:11 | [Fold 1] Stepwise step 1: testing 22 cands x 3 methods (16 workers)
  22:59:17 | [Fold 1] Step 1: + signal_X -> mean SR=0.72
  ...
  23:02:00 | [Fold 1] Stepwise complete: 5 signals (170s)
  23:02:00 | [Fold 2] IC filter: 78/152 passed
  ...
  23:10:30 | Step 7: 15 stable candidates found
  23:10:30 | Step 8: Nested stepwise on 15 candidates...
  23:10:31 | [Nested] Step 1: testing 15 cands x 4 folds x 3 methods (16 workers)
  23:10:40 | [Nested] Step 1: + signal_A -> mean SR=0.74
  ...
  23:14:00 | Step 9: Model comparison (2010-2014)
  23:14:30 | [Model] EW=0.62, IC=0.65, IV=0.60
  23:14:30 | [Model] Selected: ic_weighted (p=0.04)
  23:14:30 | >>> Stage 7: OOS Evaluation
  23:17:00 | [Stage 7] OOS Sharpe: 0.48, DSR: 0.55, Bootstrap p: 0.03
  23:17:00 | COMPLETE: 5 signals, method=ic_weighted [27.0m]
```


## Output Directory Structure

```
outputs/runs/{YYYYMMDD_HHMMSS}/
|-- config_snapshot.yaml
|-- inner_folds/
|   |-- fold_1/
|   |   |-- ic_survivors.csv
|   |   |-- dev_survivors.csv
|   |   |-- val_survivors.csv
|   |   |-- dedup_survivors.csv
|   |   |-- stepwise_history.csv
|   |   |-- signal_metrics.csv
|   |   +-- summary.json
|   |-- fold_2/ ...
|   |-- fold_3/ ...
|   +-- fold_4/ ...
|-- stability/
|   |-- survival_matrix.csv
|   |-- stability_scores.csv
|   |-- cross_fold_icir.csv
|   +-- stable_candidates.csv
|-- frozen/
|   |-- freeze_manifest.yaml      <-- git hash, all params locked
|   +-- selected_signals.json
|-- model_comparison/
|   |-- equal_metrics.csv
|   |-- ic_weighted_metrics.csv
|   |-- inverse_vol_metrics.csv
|   |-- comparison_summary.json
|   |-- dm_tests.csv
|   +-- selected_method.txt
+-- oos_evaluation/
    |-- performance.json
    |-- bias_aware.json           <-- DSR, PBO
    |-- bootstrap.json
    +-- attribution.json
```


## Key Design Principles

1. **Information flows strictly left-to-right.** Each layer only sees data from its own
   period and earlier. The OOS window (2015-2019) is never used for any decision.

2. **Method-agnostic selection.** Every candidate is evaluated under EW, IC-weighted,
   AND inverse-vol. The average score is used. This ensures signals are fundamentally
   predictive, not just good under one weighting scheme. Phase 5 model comparison
   is now a fair race.

3. **Compute-once, share everywhere.** GPU operations (IC, turnover, spread,
   neutralization) are done once globally. The 18GB precomputed_neutral dict is
   shared across all folds and workers via Linux fork copy-on-write.

4. **Parallel within, sequential across.** Within each stepwise step, 16 cores
   evaluate candidates simultaneously. Steps are sequential (step N depends on
   step N-1). Folds are sequential (to avoid nested process pool contention).

5. **Two levels of stepwise selection:**
   - Per-fold stepwise (Phase 2): individually good signals per time period
   - Nested stepwise (Phase 4): portfolio-level robustness across all periods
   A signal can pass Phase 2 in all folds but fail Phase 4 due to redundancy.

6. **Freezing protocol.** Before touching the middle layer (2010-2014), everything is
   frozen: signal set, stopping rules, cost model, neutralization config, git hash.

7. **Selection-bias correction.** DSR corrects for 770 trials tested. PBO uses
   combinatorial symmetric cross-validation across inner folds.

8. **Full traceability.** Every intermediate result saved to timestamped run directory.
   Real-time progress logged with flush to file + console.
