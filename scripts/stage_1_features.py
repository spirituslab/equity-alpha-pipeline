"""Stage 1: Compute all factor exposures and cache as parquet.

For each registered signal:
1. signal.compute(panel) → raw exposures (date x gvkey)
2. Cache to data/cache/raw_{signal_name}.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.loader import DataPanel
from src.signals.registry import SignalRegistry


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)
    registry = SignalRegistry(active_names=config.signals.active)

    print("=" * 60)
    print("  STAGE 1: FEATURE COMPUTATION")
    print("=" * 60)

    # Compute and cache each signal
    for signal in registry.get_active():
        print(f"\n  Computing {signal.name} ({signal.category})...")
        raw = signal.compute(panel)
        n_valid = raw.count().sum()
        print(f"    Shape: {raw.shape}, non-null: {n_valid:,}")

        cache_path = config.cache_path(f"raw_{signal.name}.parquet")
        raw.to_parquet(cache_path)
        print(f"    Cached: {cache_path}")

    # Also cache control variables needed for neutralization
    print("\n  Computing control variables...")

    universe = panel.get_universe()
    universe.to_parquet(config.cache_path("universe.parquet"))
    print(f"    Universe: {universe.shape}, avg size: {universe.sum(axis=1).mean():.0f}")

    mcap = panel.get_log_market_cap()
    mcap.to_parquet(config.cache_path("log_mcap.parquet"))
    print(f"    Log market cap: {mcap.shape}")

    returns = panel.get_returns()
    returns.to_parquet(config.cache_path("returns.parquet"))
    print(f"    Returns: {returns.shape}")

    print("\n  Computing rolling beta (60-month window)...")
    beta = panel.get_rolling_beta(window=60)
    beta.to_parquet(config.cache_path("beta.parquet"))
    n_valid_beta = beta.count().sum()
    print(f"    Beta: {beta.shape}, non-null: {n_valid_beta:,}")

    print("\n  Stage 1 complete.")


if __name__ == "__main__":
    main()
