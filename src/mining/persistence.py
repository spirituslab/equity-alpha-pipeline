"""Intermediate output persistence for nested validation runs.

Every artifact is saved to a timestamped run directory for full traceability.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


class RunContext:
    """Manages a timestamped output directory for a single pipeline run."""

    def __init__(self, output_dir: str, project_root: Path = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (project_root or Path.cwd()) / output_dir / "runs" / timestamp
        base.mkdir(parents=True, exist_ok=True)
        self.root = base
        self._make_subdirs()

    def _make_subdirs(self):
        for sub in [
            "inner_folds",
            "stability",
            "frozen",
            "model_comparison",
            "oos_evaluation",
        ]:
            (self.root / sub).mkdir(exist_ok=True)

    def fold_dir(self, fold_id: int) -> Path:
        d = self.root / "inner_folds" / f"fold_{fold_id}"
        d.mkdir(exist_ok=True)
        return d

    # ---- Config snapshot ----

    def save_config_snapshot(self, pipeline_config, mining_config=None):
        """Save full configuration at run start."""
        snapshot = {
            "dates": asdict(pipeline_config.dates),
            "validation": asdict(pipeline_config.validation),
            "signals": asdict(pipeline_config.signals),
            "backtest": asdict(pipeline_config.backtest),
            "optimization": asdict(pipeline_config.optimization),
            "costs": asdict(pipeline_config.costs),
        }
        if mining_config is not None:
            snapshot["mining"] = asdict(mining_config) if hasattr(mining_config, '__dataclass_fields__') else {}
        with open(self.root / "config_snapshot.yaml", "w") as f:
            yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)

    # ---- Inner fold results ----

    def save_fold_result(self, fold_id: int, fold_result):
        """Save all artifacts for one inner fold."""
        d = self.fold_dir(fold_id)

        # Signal lists at each gate
        for gate in ["ic_survivors", "dev_survivors", "val_survivors",
                     "dedup_survivors", "stepwise_selected"]:
            signals = getattr(fold_result, gate, [])
            pd.Series(signals, name="signal").to_csv(d / f"{gate}.csv", index=False)

        # Stepwise history
        if fold_result.stepwise_history:
            rows = []
            for sr in fold_result.stepwise_history:
                rows.append(asdict(sr) if hasattr(sr, '__dataclass_fields__') else vars(sr))
            pd.DataFrame(rows).to_csv(d / "stepwise_history.csv", index=False)

        # Per-signal metrics
        if fold_result.per_signal_metrics:
            pd.DataFrame(fold_result.per_signal_metrics).T.to_csv(
                d / "signal_metrics.csv"
            )

        # Fold summary
        summary = {
            "fold_id": fold_result.fold_id,
            "train_end": fold_result.train_end,
            "val_end": fold_result.val_end,
            "fold_sharpe": fold_result.fold_sharpe,
            "n_ic_survivors": len(fold_result.ic_survivors),
            "n_dev_survivors": len(fold_result.dev_survivors),
            "n_val_survivors": len(fold_result.val_survivors),
            "n_dedup_survivors": len(fold_result.dedup_survivors),
            "n_stepwise_selected": len(fold_result.stepwise_selected),
        }
        with open(d / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ---- Stability report ----

    def save_stability_report(self, report):
        """Save cross-fold stability analysis."""
        d = self.root / "stability"
        report.survival_matrix.to_csv(d / "survival_matrix.csv")
        report.stability_scores.to_csv(d / "stability_scores.csv")
        report.cross_fold_icir.to_csv(d / "cross_fold_icir.csv")
        pd.Series(report.stable_candidates, name="signal").to_csv(
            d / "stable_candidates.csv", index=False
        )
        if report.per_signal_summary is not None:
            report.per_signal_summary.to_csv(d / "per_signal_summary.csv")

    # ---- Freeze manifest ----

    def save_freeze_manifest(
        self,
        signal_names: list[str],
        pipeline_config,
        mining_config=None,
    ):
        """Save the frozen specification before model comparison."""
        manifest = {
            "frozen_at": datetime.now().isoformat(),
            "signal_names": signal_names,
            "n_signals": len(signal_names),
            "validation_config": asdict(pipeline_config.validation),
            "cost_config": asdict(pipeline_config.costs),
            "neutralization": {
                "sector": pipeline_config.signals.neutralize_sector,
                "size": pipeline_config.signals.neutralize_size,
                "beta": pipeline_config.signals.neutralize_beta,
            },
        }
        # Git commit hash
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=pipeline_config.project_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            manifest["git_commit"] = git_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            manifest["git_commit"] = "unknown"

        with open(self.root / "frozen" / "freeze_manifest.yaml", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        with open(self.root / "frozen" / "selected_signals.json", "w") as f:
            json.dump(signal_names, f, indent=2)

    # ---- Model comparison ----

    def save_model_comparison(self, comparison_result):
        """Save middle-layer model comparison results."""
        d = self.root / "model_comparison"

        # Per-method metrics
        for method, metrics in comparison_result.method_results.items():
            if isinstance(metrics, dict):
                pd.Series(metrics).to_csv(d / f"{method}_metrics.csv")
            elif isinstance(metrics, pd.Series):
                metrics.to_csv(d / f"{method}_metrics.csv")

        # Summary
        summary = {
            "selected_method": comparison_result.selected_method,
            "method_sharpes": comparison_result.method_sharpes,
        }
        with open(d / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # DM tests
        if comparison_result.dm_tests:
            rows = []
            for (m1, m2), result in comparison_result.dm_tests.items():
                row = {"method_1": m1, "method_2": m2}
                row.update(result)
                rows.append(row)
            pd.DataFrame(rows).to_csv(d / "dm_tests.csv", index=False)

        with open(d / "selected_method.txt", "w") as f:
            f.write(comparison_result.selected_method)

    # ---- OOS evaluation ----

    def save_oos_evaluation(self, metrics: dict):
        """Save final OOS evaluation results."""
        d = self.root / "oos_evaluation"
        for key, value in metrics.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(d / f"{key}.csv")
            elif isinstance(value, pd.Series):
                value.to_csv(d / f"{key}.csv")
            elif isinstance(value, dict):
                with open(d / f"{key}.json", "w") as f:
                    json.dump(value, f, indent=2, default=str)
            else:
                with open(d / f"{key}.txt", "w") as f:
                    f.write(str(value))
