"""Mining orchestrator: enumerate → compute → evaluate → filter → dedup → codegen.

Supports both single-split (legacy) and nested chronological validation modes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.mining.config import MiningConfig
from src.mining.enumeration import enumerate_candidates, CandidateSpec
from src.mining.compute import compute_all_candidates
from src.mining.evaluate import quick_evaluate, validate_candidate, batch_evaluate_gpu, EvalResult
from src.mining.filter import filter_dev_period, filter_val_period
from src.signals.zscore import standardize_signal
from src.mining.deduplicate import deduplicate
from src.mining.codegen import generate_factor_file


def run_mining(pipeline_config: PipelineConfig, mining_config: MiningConfig) -> None:
    """Full mining pipeline."""
    panel = DataPanel(pipeline_config)
    returns = panel.get_returns()
    universe = panel.get_universe()

    print("=" * 70)
    print("  SIGNAL MINING MACHINE")
    print("=" * 70)

    # ---- 1. Enumerate ----
    print("\n  Step 1: Enumerating candidates...")
    specs = enumerate_candidates(mining_config)
    specs_by_name = {s.name: s for s in specs}
    print(f"    Generated {len(specs)} candidate specifications")

    # Category breakdown
    cats = {}
    for s in specs:
        cats[s.category] = cats.get(s.category, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"      {cat}: {count}")

    # ---- 2. Compute ----
    print("\n  Step 2: Computing candidate signals...")
    raw_signals = compute_all_candidates(panel, specs, show_progress=True)

    # ---- 3. Cascaded evaluation: IC first (GPU), then turnover+spread only for survivors ----
    from src.gpu.backend import GPU_AVAILABLE
    from src.gpu.ic_batch import batch_compute_ic
    from src.analytics.ic import compute_ic_series, ic_summary as _ic_summary
    from src.signals.report_card import _compute_signal_turnover, _compute_decile_spread

    # Step 3a: Standardize all signals
    print("\n  Step 3a: Standardizing signals...")
    z_signals = {}
    for name, raw in raw_signals.items():
        z_signals[name] = standardize_signal(raw, winsorize_pct=0.01)

    # Step 3b: GPU batch IC (fast, ~1 min for 770 signals)
    print(f"\n  Step 3b: GPU batch IC evaluation ({len(z_signals)} signals)...")
    if GPU_AVAILABLE and len(z_signals) > 20:
        ic_dict = batch_compute_ic(z_signals, returns, universe, end_date=pipeline_config.dates.dev_end)
    else:
        ic_dict = {}
        for name in tqdm(z_signals, desc="IC eval"):
            ic_dict[name] = compute_ic_series(z_signals[name], returns, universe, end_date=pipeline_config.dates.dev_end)

    # Build IC-only evaluations
    evaluations = {}
    for name in raw_signals:
        ev = EvalResult(name=name)
        if name in ic_dict and len(ic_dict[name]) > 0:
            stats = _ic_summary(ic_dict[name])
            ev.dev_ic = stats.get("Mean IC", np.nan)
            ev.dev_icir = stats.get("ICIR", np.nan)
            ev.dev_hit_rate = stats.get("Hit Rate", np.nan)
            ev.dev_t_stat = stats.get("t-stat", np.nan)
            ev.n_months_dev = stats.get("N Months", 0)
        evaluations[name] = ev

    # Step 3c: Filter by IC first (cheap filter, removes ~80% of candidates)
    ic_threshold = mining_config.min_icir_dev
    ic_survivors = {n: ev for n, ev in evaluations.items()
                    if not np.isnan(ev.dev_icir) and abs(ev.dev_icir) >= ic_threshold}
    print(f"\n  Step 3c: IC filter: {len(ic_survivors)} / {len(evaluations)} passed (|ICIR| >= {ic_threshold})")

    # Step 3d: Compute turnover + decile spread ONLY for IC survivors
    print(f"\n  Step 3d: Computing turnover + spread for {len(ic_survivors)} IC survivors...")
    for i, (name, ev) in enumerate(ic_survivors.items()):
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(ic_survivors)}]...")
        ev.turnover = _compute_signal_turnover(z_signals[name])
        spread = _compute_decile_spread(z_signals[name], returns, universe, end_date=pipeline_config.dates.dev_end)
        ev.dev_spread = spread.get("spread", np.nan)
        ev.dev_spread_t = spread.get("spread_t", np.nan)

    # ---- 4. Filter remaining thresholds (turnover, hit rate, spread) ----
    print("\n  Step 4: Filtering by remaining thresholds...")
    dev_survivors = filter_dev_period(ic_survivors, mining_config)
    print(f"    {len(dev_survivors)} / {len(ic_survivors)} passed all dev filters")
    print(f"    Thresholds: |ICIR|>{mining_config.min_icir_dev}, "
          f"HR>{mining_config.min_hit_rate}, "
          f"TO<{mining_config.max_turnover}, "
          f"|spread_t|>{mining_config.min_spread_t}")

    if not dev_survivors:
        print("\n  No candidates passed dev filters. Loosening thresholds...")
        mining_config.min_icir_dev = 0.10
        mining_config.min_hit_rate = 0.50
        mining_config.min_spread_t = 1.5
        dev_survivors = filter_dev_period(ic_survivors, mining_config)
        print(f"    {len(dev_survivors)} passed with loosened thresholds")

    # ---- 5. Validate survivors on holdout period ----
    print("\n  Step 5: Validating on holdout period (2005-2014)...")
    for name, ev in dev_survivors.items():
        val_ic, val_icir = validate_candidate(
            name, raw_signals[name], returns, universe,
            val_start=pipeline_config.dates.dev_end,
            val_end=pipeline_config.dates.val_end,
        )
        ev.val_ic = val_ic
        ev.val_icir = val_icir

    val_survivors = filter_val_period(dev_survivors, mining_config)
    print(f"    {len(val_survivors)} / {len(dev_survivors)} confirmed on validation")

    # ---- 6. Deduplicate vs existing signals ----
    print("\n  Step 6: Deduplicating against existing signals...")
    existing = _load_existing_signals(pipeline_config)
    print(f"    {len(existing)} existing signals loaded")

    # Standardize survivors for correlation comparison
    std_signals = {}
    for name in val_survivors:
        std_signals[name] = standardize_signal(raw_signals[name], winsorize_pct=0.01)
    std_existing = {}
    for name, sig in existing.items():
        std_existing[name] = standardize_signal(sig, winsorize_pct=0.01)

    accepted_names = deduplicate(
        val_survivors, std_signals, std_existing,
        max_corr=mining_config.max_correlation,
    )
    print(f"    {len(accepted_names)} non-redundant signals after dedup")

    # ---- 7. Print results ----
    print(f"\n{'='*70}")
    print(f"  MINING RESULTS: {len(accepted_names)} new signals discovered")
    print(f"{'='*70}")

    if accepted_names:
        print(f"\n  {'Name':35s} {'Dev ICIR':>9s} {'Val ICIR':>9s} {'HR':>6s} {'TO':>6s} {'Spr t':>7s} {'Cat':>10s}")
        print(f"  {'-'*85}")
        for name in sorted(accepted_names, key=lambda n: abs(val_survivors[n].dev_icir), reverse=True):
            ev = val_survivors[name]
            print(f"  {name:35s} {ev.dev_icir:>+9.4f} {ev.val_icir:>+9.4f} "
                  f"{ev.dev_hit_rate:>6.3f} {ev.turnover:>6.3f} {ev.dev_spread_t:>+7.2f} "
                  f"{specs_by_name[name].category:>10s}")

    # ---- 8. Generate code ----
    output_dir = Path(pipeline_config.project_root) / mining_config.factor_dir
    print(f"\n  Step 8: Generating Factor subclass files in {output_dir}...")
    for name in accepted_names:
        spec = specs_by_name[name]
        filepath = generate_factor_file(spec, val_survivors[name], output_dir)
        print(f"    Generated: {filepath.name}")

    # ---- 9. Save full results CSV ----
    results_path = Path(pipeline_config.project_root) / mining_config.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _save_results_csv(evaluations, specs_by_name, results_path)
    print(f"\n  Full results saved to {results_path}")

    # ---- 10. Print top 20 overall ----
    print(f"\n{'='*70}")
    print(f"  TOP 20 CANDIDATES (by |Dev ICIR|, all candidates)")
    print(f"{'='*70}")

    ranked = sorted(evaluations.values(), key=lambda e: abs(e.dev_icir) if not np.isnan(e.dev_icir) else 0, reverse=True)
    print(f"\n  {'Name':35s} {'Dev ICIR':>9s} {'HR':>6s} {'TO':>6s} {'Spr t':>7s} {'Dev?':>5s} {'Val?':>5s} {'Dup?':>5s}")
    print(f"  {'-'*85}")
    for ev in ranked[:20]:
        d = "Y" if ev.passed_dev else ""
        v = "Y" if ev.passed_val else ""
        u = "Y" if ev.passed_dedup else ""
        print(f"  {ev.name:35s} {ev.dev_icir:>+9.4f} {ev.dev_hit_rate:>6.3f} "
              f"{ev.turnover:>6.3f} {ev.dev_spread_t:>+7.2f} {d:>5s} {v:>5s} {u:>5s}")

    print(f"\n{'='*70}")
    print(f"  Mining complete. Add survivors to src/signals/registry.py and config/pipeline.yaml")
    print(f"{'='*70}")


def _load_existing_signals(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load existing cached raw signals.

    Returns empty dict when signals.active is empty (all signals come from mining).
    """
    existing = {}
    for name in config.signals.active:
        cache_file = config.cache_path(f"raw_{name}.parquet")
        if cache_file.exists():
            existing[name] = pd.read_parquet(cache_file)
    return existing


def _save_results_csv(
    evaluations: dict[str, EvalResult],
    specs_by_name: dict[str, CandidateSpec],
    path: Path,
) -> None:
    """Save all evaluation results to CSV."""
    rows = []
    for name, ev in evaluations.items():
        spec = specs_by_name.get(name)
        rows.append({
            "name": name,
            "category": spec.category if spec else "",
            "transform": spec.transform.value if spec else "",
            "field_a": spec.field_a if spec else "",
            "field_b": spec.field_b if spec else "",
            "dev_ic": ev.dev_ic,
            "dev_icir": ev.dev_icir,
            "dev_hit_rate": ev.dev_hit_rate,
            "dev_t_stat": ev.dev_t_stat,
            "dev_spread": ev.dev_spread,
            "dev_spread_t": ev.dev_spread_t,
            "turnover": ev.turnover,
            "val_ic": ev.val_ic,
            "val_icir": ev.val_icir,
            "n_months_dev": ev.n_months_dev,
            "passed_dev": ev.passed_dev,
            "passed_val": ev.passed_val,
            "passed_dedup": ev.passed_dedup,
        })
    df = pd.DataFrame(rows).sort_values("dev_icir", key=abs, ascending=False)
    df.to_csv(path, index=False)


def run_mining_nested(
    pipeline_config: PipelineConfig,
    mining_config: MiningConfig,
    run_ctx,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    sectors: pd.DataFrame,
    log_mcap: pd.DataFrame,
    beta: pd.DataFrame,
    factor_returns: pd.DataFrame,
    rf: pd.Series,
    logger=None,
) -> tuple[list[str], str, dict[str, pd.DataFrame]]:
    """Nested validation mining pipeline.

    Returns:
        (signal_names, selected_method, precomputed_neutral)
    """
    from src.gpu.backend import GPU_AVAILABLE
    from src.gpu.ic_batch import batch_compute_ic
    from src.gpu.turnover_batch import batch_compute_turnover
    from src.gpu.spread_batch import batch_compute_decile_spread
    from src.gpu.neutralize_batch import ProjectionCache
    from src.analytics.ic import ic_summary as _ic_summary
    from src.mining.inner_folds import InnerFoldRunner, FoldResult
    from src.mining.stability import StabilityTracker
    from src.mining.stepwise import forward_stepwise_nested
    from src.mining.model_comparison import ModelComparison

    vc = pipeline_config.validation
    panel = DataPanel(pipeline_config)

    import time as _time
    def _log(msg):
        if logger:
            logger.progress(msg)
        else:
            print(msg, flush=True)

    def _logsub(parent, detail):
        if logger:
            logger.substep(parent, detail)
        else:
            print(f"  [{parent}] {detail}", flush=True)

    _log("=" * 70)
    _log("NESTED VALIDATION MINING PIPELINE")
    _log(f"Inner: {vc.inner_start}→{vc.inner_end}, Middle: {vc.middle_start}→{vc.middle_end}, OOS: {vc.oos_start}→")
    _log(f"Folds: {len(vc.inner_folds)}")

    # ---- 1. Enumerate + compute (same as single-split) ----
    t0 = _time.time()
    _log("Step 1: Enumerating candidates...")
    specs = enumerate_candidates(mining_config)
    specs_by_name = {s.name: s for s in specs}
    _log(f"Step 1: Generated {len(specs)} candidate specifications ({_time.time()-t0:.0f}s)")

    t0 = _time.time()
    _log("Step 2: Computing candidate signals...")
    raw_signals = compute_all_candidates(panel, specs, show_progress=True)
    _log(f"Step 2: Computed {len(raw_signals)} signals ({_time.time()-t0:.0f}s)")

    # ---- 3. Standardize all signals ----
    t0 = _time.time()
    _log("Step 3: Standardizing signals...")
    z_signals = {}
    for name, raw in raw_signals.items():
        z_signals[name] = standardize_signal(raw, winsorize_pct=0.01)
    _log(f"Step 3: Standardized {len(z_signals)} signals ({_time.time()-t0:.0f}s)")

    # ---- 4. Global precomputation (GPU) ----
    # 4a. Build projection cache once
    t0 = _time.time()
    _log("Step 4a: Building projection matrix cache...")
    proj_cache = ProjectionCache()
    proj_cache.build(
        sector_labels=sectors, log_mcap=log_mcap, beta=beta,
        neutralize_sector=True, neutralize_size=True, neutralize_beta=True,
    )
    _log(f"Step 4a: Cached {len(proj_cache.projections)} projection matrices ({_time.time()-t0:.0f}s)")

    # 4b. Loose IC filter on full inner period to reduce candidates
    t0 = _time.time()
    _log(f"Step 4b: GPU IC on full inner period ({vc.inner_start}→{vc.inner_end})...")
    if GPU_AVAILABLE and len(z_signals) > 20:
        global_ic = batch_compute_ic(z_signals, returns, universe, end_date=vc.inner_end)
    else:
        from src.analytics.ic import compute_ic_series
        global_ic = {name: compute_ic_series(z_signals[name], returns, universe, end_date=vc.inner_end)
                     for name in tqdm(z_signals, desc="IC eval")}

    # Keep signals with |ICIR| > loose threshold (half of dev threshold)
    loose_threshold = mining_config.min_icir_dev
    ic_candidates = []
    for name in raw_signals:
        if name in global_ic and len(global_ic[name]) > 0:
            stats = _ic_summary(global_ic[name])
            icir = stats.get("ICIR", 0)
            if abs(icir) >= loose_threshold:
                ic_candidates.append(name)
    _log(f"Step 4b: Loose IC filter: {len(ic_candidates)} / {len(raw_signals)} passed (|ICIR| >= {loose_threshold}) ({_time.time()-t0:.0f}s)")

    # 4c. GPU batch turnover + spread for IC candidates
    ic_z = {n: z_signals[n] for n in ic_candidates}
    t0 = _time.time()
    _log(f"Step 4c: GPU batch turnover + spread for {len(ic_candidates)} candidates...")
    turnover_dict = batch_compute_turnover(ic_z, universe, end_date=vc.inner_end)
    spread_dict = batch_compute_decile_spread(ic_z, returns, universe, end_date=vc.inner_end)

    _log(f"Step 4c: Turnover + spread done ({_time.time()-t0:.0f}s)")

    # 4d. Neutralize ALL IC candidates once
    t0 = _time.time()
    _log(f"Step 4d: Neutralizing {len(ic_candidates)} candidates...")
    precomputed_neutral = {}
    for i, name in enumerate(ic_candidates):
        z = z_signals[name]
        if proj_cache._built:
            if GPU_AVAILABLE:
                n = proj_cache.neutralize_fast_gpu(z)
            else:
                n = proj_cache.neutralize_fast(z)
        else:
            from src.signals.neutralize import neutralize_signal
            n = neutralize_signal(z, sector_labels=sectors, log_mcap=log_mcap, beta=beta,
                                  neutralize_sector=True, neutralize_size=True, neutralize_beta=True)
        precomputed_neutral[name] = n
        if (i + 1) % 10 == 0:
            _logsub("Step 4d", f"[{i+1}/{len(ic_candidates)}] neutralized")
    _log(f"Step 4d: All {len(precomputed_neutral)} signals neutralized ({_time.time()-t0:.0f}s)")

    # ---- 5. Per-fold IC computation (GPU, sequential) ----
    t0 = _time.time()
    _log("Step 5: Per-fold GPU IC computation...")
    fold_ic_dicts = {}
    for i, fd in enumerate(vc.inner_folds):
        t1 = _time.time()
        _logsub("Step 5", f"Fold {i+1}: IC up to {fd.train_end}...")
        if GPU_AVAILABLE and len(ic_z) > 20:
            fold_ic = batch_compute_ic(ic_z, returns, universe, end_date=fd.train_end)
        else:
            from src.analytics.ic import compute_ic_series
            fold_ic = {name: compute_ic_series(ic_z[name], returns, universe, end_date=fd.train_end)
                       for name in ic_z}
        fold_ic_dicts[i] = fold_ic
        _logsub("Step 5", f"Fold {i+1} IC done ({_time.time()-t1:.0f}s)")
    _log(f"Step 5: All fold ICs computed ({_time.time()-t0:.0f}s)")

    # ---- 6. Run inner folds ----
    t0 = _time.time()
    _log(f"Step 6: Running {len(vc.inner_folds)} inner folds...")
    fold_results = []
    ic_raw = {n: raw_signals[n] for n in ic_candidates}

    for i, fd in enumerate(vc.inner_folds):
        runner = InnerFoldRunner(
            fold_id=i + 1,
            fold_def=fd,
            raw_signals=ic_raw,
            z_signals=ic_z,
            ic_dict=fold_ic_dicts[i],
            turnover_dict=turnover_dict,
            spread_dict=spread_dict,
            returns=returns,
            universe=universe,
            factor_returns=factor_returns,
            rf=rf,
            pipeline_config=pipeline_config,
            mining_config=mining_config,
            precomputed_neutral=precomputed_neutral,
            validation_config=vc,
        )
        result = runner.run()
        fold_results.append(result)
        run_ctx.save_fold_result(i + 1, result)

    _log(f"Step 6: All folds complete ({_time.time()-t0:.0f}s)")

    # ---- 7. Cross-fold stability ----
    _log("Step 7: Cross-fold stability analysis...")
    tracker = StabilityTracker(fold_results, vc)
    stability = tracker.compute()
    run_ctx.save_stability_report(stability)

    _log(f"Step 7: {len(stability.stable_candidates)} stable candidates found")
    if stability.stable_candidates:
        for sig in stability.stable_candidates[:10]:
            score = stability.stability_scores.get(sig, 0)
            _logsub("Stability", f"{sig:35s} score={score:.2f}")

    if not stability.stable_candidates:
        _log("WARNING: No stable candidates found. Lowering threshold...")
        # Fallback: take signals surviving at least 1 fold
        vc_relaxed = type(vc)(**{**vc.__dict__, 'min_fold_survival': 1, 'stability_threshold': 0.25})
        tracker_relaxed = StabilityTracker(fold_results, vc_relaxed)
        stability = tracker_relaxed.compute()
        _log(f"Step 7: Relaxed → {len(stability.stable_candidates)} stable candidates")

    # ---- 8. Nested stepwise on stable candidates ----
    t0 = _time.time()
    _log(f"Step 8: Nested forward stepwise on {len(stability.stable_candidates)} stable candidates...")
    nested_history = forward_stepwise_nested(
        stable_candidates=stability.stable_candidates,
        precomputed_neutral=precomputed_neutral,
        returns=returns,
        universe=universe,
        config=pipeline_config,
        fold_defs=vc.inner_folds,
        validation_config=vc,
        max_signals=15,
    )

    if nested_history:
        best = nested_history[-1]
        selected_signals = best.signal_set
    else:
        selected_signals = stability.stable_candidates[:5]
        _log(f"Step 8: Stepwise returned empty — using top {len(selected_signals)} stable candidates")

    # ---- 9. Freeze and model comparison ----
    run_ctx.save_freeze_manifest(selected_signals, pipeline_config, mining_config)

    _log(f"Step 8: Nested stepwise complete ({_time.time()-t0:.0f}s)")
    _log(f"Step 9: Model comparison on middle layer ({vc.middle_start}→{vc.middle_end})...")
    comparison = ModelComparison(
        frozen_signals=selected_signals,
        precomputed_neutral=precomputed_neutral,
        returns=returns,
        universe=universe,
        factor_returns=factor_returns,
        rf=rf,
        config=pipeline_config,
    )
    comparison_result = comparison.run()
    run_ctx.save_model_comparison(comparison_result)

    selected_method = comparison_result.selected_method
    _log(f"Step 9: Selected method={selected_method}, Sharpes={comparison_result.method_sharpes}")
    _log(f"Mining complete: {len(selected_signals)} signals, method={selected_method}")

    return selected_signals, selected_method, precomputed_neutral
