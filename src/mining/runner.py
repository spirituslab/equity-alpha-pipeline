"""Mining orchestrator: enumerate → compute → evaluate → filter → dedup → codegen."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.mining.config import MiningConfig
from src.mining.enumeration import enumerate_candidates, CandidateSpec
from src.mining.compute import compute_all_candidates
from src.mining.evaluate import quick_evaluate, validate_candidate, EvalResult
from src.mining.filter import filter_dev_period, filter_val_period
from src.mining.deduplicate import deduplicate
from src.mining.codegen import generate_factor_file
from src.signals.zscore import standardize_signal


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

    # ---- 3. Evaluate on dev period ----
    print("\n  Step 3: Evaluating on development period (1975-2004)...")
    evaluations = {}
    for name in tqdm(raw_signals, desc="Evaluating"):
        evaluations[name] = quick_evaluate(
            name, raw_signals[name], returns, universe,
            dev_end=pipeline_config.dates.dev_end,
            val_end=pipeline_config.dates.val_end,
        )

    # ---- 4. Filter by dev thresholds ----
    print("\n  Step 4: Filtering by dev-period thresholds...")
    dev_survivors = filter_dev_period(evaluations, mining_config)
    print(f"    {len(dev_survivors)} / {len(evaluations)} passed dev filters")
    print(f"    Thresholds: |ICIR|>{mining_config.min_icir_dev}, "
          f"HR>{mining_config.min_hit_rate}, "
          f"TO<{mining_config.max_turnover}, "
          f"|spread_t|>{mining_config.min_spread_t}")

    if not dev_survivors:
        print("\n  No candidates passed dev filters. Loosening thresholds...")
        mining_config.min_icir_dev = 0.10
        mining_config.min_hit_rate = 0.50
        mining_config.min_spread_t = 1.5
        dev_survivors = filter_dev_period(evaluations, mining_config)
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
    """Load existing cached raw signals."""
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
