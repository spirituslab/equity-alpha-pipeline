"""Run full pipeline: features → signals → backtest → ML → robustness."""

import subprocess
import sys
import time
from pathlib import Path

STAGES = [
    ("Stage 1: Features", "scripts/stage_1_features.py"),
    ("Stage 2: Signals", "scripts/stage_2_signals.py"),
    ("Stage 3: Backtest", "scripts/stage_3_backtest.py"),
    ("Stage 4: ML Extension", "scripts/stage_4_ml.py"),
    ("Stage 5: Robustness", "scripts/stage_5_robustness.py"),
]


def main():
    print("=" * 70)
    print("  EQUITY ALPHA PIPELINE — FULL RUN")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    failed = []

    for name, script in STAGES:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}\n")

        start = time.time()
        result = subprocess.run(
            [sys.executable, str(project_root / script)],
            cwd=str(project_root),
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n  *** {name} FAILED (exit code {result.returncode}) ***")
            failed.append(name)
        else:
            print(f"\n  {name} completed in {elapsed:.1f}s")

    print(f"\n\n{'='*70}")
    if failed:
        print(f"  COMPLETED WITH FAILURES: {', '.join(failed)}")
    else:
        print(f"  ALL STAGES COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
