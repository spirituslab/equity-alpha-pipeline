"""Stage 2: Standardize, neutralize, and combine signals.

1. Load cached raw signals from Stage 1
2. Winsorize + z-score each signal cross-sectionally
3. Neutralize against sector / size / beta
4. Combine into composite alpha score
5. Generate signal report cards
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.sectors import assign_sectors, load_sector_mapping
from src.data.loader import DataPanel
from src.signals.registry import SignalRegistry
from src.signals.zscore import standardize_signal
from src.signals.neutralize import neutralize_all_signals, verify_neutralization
from src.signals.combine import combine_signals
from src.signals.report_card import signal_report_card, print_report_card


def main():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")
    panel = DataPanel(config)
    registry = SignalRegistry(active_names=config.signals.active)

    print("=" * 60)
    print("  STAGE 2: SIGNAL PROCESSING")
    print("=" * 60)

    # Load cached data from Stage 1
    print("\n  Loading cached data...")
    universe = pd.read_parquet(config.cache_path("universe.parquet"))
    log_mcap = pd.read_parquet(config.cache_path("log_mcap.parquet"))
    beta = pd.read_parquet(config.cache_path("beta.parquet"))
    returns = pd.read_parquet(config.cache_path("returns.parquet"))

    # Load sector labels
    print("  Loading sector labels...")
    try:
        sectors = assign_sectors(panel, method="static")
        print(f"    Sectors shape: {sectors.shape}")
    except ValueError as e:
        print(f"    WARNING: {e}")
        print("    Proceeding without sector neutralization.")
        sectors = None
        config.signals.neutralize_sector = False

    # Step 1: Load and standardize raw signals
    print("\n  Standardizing signals...")
    raw_signals = {}
    z_signals = {}
    for signal in registry.get_active():
        cache_file = config.cache_path(f"raw_{signal.name}.parquet")
        raw = pd.read_parquet(cache_file)
        raw_signals[signal.name] = raw

        z = standardize_signal(raw, winsorize_pct=config.signals.winsorize_pct)
        z_signals[signal.name] = z
        print(f"    {signal.name}: z-scored, non-null={z.count().sum():,}")

        z.to_parquet(config.cache_path(f"zscore_{signal.name}.parquet"))

    # Step 2: Neutralize
    print("\n  Neutralizing signals against sector/size/beta...")
    neutral_signals = neutralize_all_signals(
        z_signals,
        sector_labels=sectors if sectors is not None else pd.DataFrame(),
        log_mcap=log_mcap,
        beta=beta,
        neutralize_sector=config.signals.neutralize_sector,
        neutralize_size=config.signals.neutralize_size,
        neutralize_beta=config.signals.neutralize_beta,
    )

    for name, sig in neutral_signals.items():
        sig.to_parquet(config.cache_path(f"neutral_{name}.parquet"))

    # Step 3: Verify neutralization
    print("\n  Verifying neutralization quality...")
    for name, sig in neutral_signals.items():
        quality = verify_neutralization(sig, sectors if sectors is not None else pd.DataFrame(), log_mcap, beta)
        size_corr = quality.get("avg_abs_corr_size", float("nan"))
        beta_corr = quality.get("avg_abs_corr_beta", float("nan"))
        sec_corr = quality.get("avg_abs_corr_sector", float("nan"))
        status = "OK" if all(v < 0.05 for v in [size_corr, beta_corr] if not pd.isna(v)) else "CHECK"
        print(f"    {name:25s}: size={size_corr:.4f}, beta={beta_corr:.4f}, sector={sec_corr:.4f} [{status}]")

    # Step 4: Combine signals
    print(f"\n  Combining signals (method={config.signals.combination_method})...")
    composite = combine_signals(
        neutral_signals,
        method=config.signals.combination_method,
        returns=returns,
        universe=universe,
        lookback=config.backtest.lookback_signal,
    )
    composite.to_parquet(config.cache_path("composite_alpha.parquet"))
    print(f"    Composite alpha shape: {composite.shape}, non-null: {composite.count().sum():,}")

    # Step 5: Signal report cards
    print("\n  Generating signal report cards...")
    for name, sig in neutral_signals.items():
        others = {n: s for n, s in neutral_signals.items() if n != name}
        report = signal_report_card(
            name, sig, returns, universe,
            other_signals=others,
            end_date=config.dates.val_end,
        )
        print_report_card(report)

    print("\n  Stage 2 complete.")


if __name__ == "__main__":
    main()
