"""Stage 7: Forward stepwise signal selection at the portfolio level.

Tests all 40 candidates (6 original + 34 mined) and finds the optimal
combination by actually running backtests at each step.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.data.french import load_french_factors
from src.data.sectors import assign_sectors
from src.mining.stepwise import forward_stepwise_selection


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)

    print("=" * 80)
    print("  STAGE 7: FORWARD STEPWISE SIGNAL SELECTION")
    print("=" * 80)

    # Load control variables
    print("\n  Loading data...")
    returns = pd.read_parquet(config.cache_path("returns.parquet"))
    universe = pd.read_parquet(config.cache_path("universe.parquet"))
    log_mcap = pd.read_parquet(config.cache_path("log_mcap.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))

    try:
        sectors = assign_sectors(panel, method="static")
    except ValueError:
        sectors = pd.DataFrame()

    ff_factors = load_french_factors(start="1962-01", end="2020-01")
    rf = ff_factors["RF"]
    ff_no_rf = ff_factors.drop(columns=["RF"])

    # Load ALL candidate signals (6 original + 34 mined)
    print("  Loading all candidate signals...")
    candidates = {}

    # 6 original
    original_names = ["momentum_12_2", "st_reversal", "roe", "asset_growth",
                      "gross_profitability", "accrual_ratio"]
    for name in original_names:
        cache_file = config.cache_path(f"raw_{name}.parquet")
        if cache_file.exists():
            candidates[name] = pd.read_parquet(cache_file)
            print(f"    {name}: loaded from cache")

    # 34 mined — compute if not cached
    mined_dir = Path("src/factors/mined")
    for py_file in sorted(mined_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        signal_name = py_file.stem

        cache_file = config.cache_path(f"raw_{signal_name}.parquet")
        if cache_file.exists():
            candidates[signal_name] = pd.read_parquet(cache_file)
        else:
            # Import and compute
            try:
                module_path = f"src.factors.mined.{signal_name}"
                mod = __import__(module_path, fromlist=[""])
                # Find the Factor subclass in the module
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if isinstance(cls, type) and hasattr(cls, "compute") and hasattr(cls, "name") and cls.name == signal_name:
                        print(f"    {signal_name}: computing...")
                        raw = cls().compute(panel)
                        raw.to_parquet(cache_file)
                        candidates[signal_name] = raw
                        break
            except Exception as e:
                print(f"    {signal_name}: FAILED ({e})")

    print(f"\n  Total candidates: {len(candidates)}")

    # Run stepwise selection
    history = forward_stepwise_selection(
        candidate_signals=candidates,
        returns=returns,
        universe=universe,
        sector_labels=sectors,
        log_mcap=log_mcap,
        beta=beta,
        factor_returns=ff_no_rf,
        rf=rf,
        config=config,
        eval_metric="full_sharpe",
        max_signals=15,
        min_improvement=0.01,
    )

    # Save results
    if history:
        rows = [{
            "step": r.step,
            "added_signal": r.added_signal,
            "n_signals": r.n_signals,
            "full_sharpe": r.full_sharpe,
            "oos_sharpe": r.oos_sharpe,
            "ff_alpha": r.ff_alpha,
            "ff_alpha_t": r.ff_alpha_t,
            "improvement": r.improvement,
            "signal_set": "|".join(r.signal_set),
        } for r in history]
        df = pd.DataFrame(rows)
        output_path = config.cache_path("stepwise_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\n  Results saved to {output_path}")

    print(f"\n  Stage 7 complete.")


if __name__ == "__main__":
    main()
