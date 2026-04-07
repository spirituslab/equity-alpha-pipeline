"""Cross-fold stability tracking for nested validation.

Aggregates results across inner folds to identify signals that
survive consistently, not just in one lucky period.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import NestedValidationConfig


@dataclass
class StabilityReport:
    survival_matrix: pd.DataFrame      # signals × folds, bool at each gate
    stability_scores: pd.Series        # per-signal float [0, 1]
    cross_fold_icir: pd.DataFrame      # signals × folds, ICIR values
    stable_candidates: list[str]       # signals meeting thresholds
    per_signal_summary: pd.DataFrame   # mean/median ICIR, survival count, etc.


class StabilityTracker:
    """Aggregate inner fold results to find stable signals."""

    def __init__(self, fold_results: list, config: NestedValidationConfig):
        self.fold_results = fold_results
        self.config = config
        self.n_folds = len(fold_results)

    def compute(self) -> StabilityReport:
        # Collect all signal names seen across folds
        all_signals = set()
        for fr in self.fold_results:
            all_signals.update(fr.ic_survivors)
            all_signals.update(fr.dev_survivors)
            all_signals.update(fr.val_survivors)
            all_signals.update(fr.dedup_survivors)
            all_signals.update(fr.stepwise_selected)
        all_signals = sorted(all_signals)

        # Build survival matrix: signals × folds
        # For each gate, track which signals passed
        gates = ["ic_survivors", "dev_survivors", "val_survivors",
                 "dedup_survivors", "stepwise_selected"]
        survival_data = {}
        for gate in gates:
            for i, fr in enumerate(self.fold_results):
                col = f"fold_{fr.fold_id}_{gate}"
                passed = set(getattr(fr, gate, []))
                survival_data[col] = {s: s in passed for s in all_signals}

        survival_matrix = pd.DataFrame(survival_data, index=all_signals)

        # Stability score: fraction of folds where signal was selected in stepwise
        stepwise_cols = [c for c in survival_matrix.columns if "stepwise_selected" in c]
        stability_scores = survival_matrix[stepwise_cols].sum(axis=1) / self.n_folds
        stability_scores = stability_scores.sort_values(ascending=False)

        # Cross-fold ICIR: extract from per_signal_metrics
        icir_data = {}
        for fr in self.fold_results:
            fold_icir = {}
            for sig_name, metrics in (fr.per_signal_metrics or {}).items():
                fold_icir[sig_name] = metrics.get("icir_train", np.nan)
            icir_data[f"fold_{fr.fold_id}"] = fold_icir
        cross_fold_icir = pd.DataFrame(icir_data)
        cross_fold_icir = cross_fold_icir.reindex(all_signals)

        # Per-signal summary
        summary_rows = []
        for sig in all_signals:
            # Count folds survived at each gate
            ic_count = sum(
                1 for fr in self.fold_results if sig in fr.ic_survivors
            )
            dev_count = sum(
                1 for fr in self.fold_results if sig in fr.dev_survivors
            )
            val_count = sum(
                1 for fr in self.fold_results if sig in fr.val_survivors
            )
            dedup_count = sum(
                1 for fr in self.fold_results if sig in fr.dedup_survivors
            )
            stepwise_count = sum(
                1 for fr in self.fold_results if sig in fr.stepwise_selected
            )

            # ICIR stats across folds
            icirs = cross_fold_icir.loc[sig].dropna().values
            mean_icir = float(np.mean(icirs)) if len(icirs) > 0 else np.nan
            median_icir = float(np.median(icirs)) if len(icirs) > 0 else np.nan
            icir_positive_frac = float((icirs > 0).mean()) if len(icirs) > 0 else 0

            summary_rows.append({
                "signal": sig,
                "ic_folds": ic_count,
                "dev_folds": dev_count,
                "val_folds": val_count,
                "dedup_folds": dedup_count,
                "stepwise_folds": stepwise_count,
                "stability_score": stability_scores.get(sig, 0),
                "mean_icir": mean_icir,
                "median_icir": median_icir,
                "icir_positive_frac": icir_positive_frac,
            })

        per_signal_summary = pd.DataFrame(summary_rows).set_index("signal")
        per_signal_summary = per_signal_summary.sort_values(
            "stability_score", ascending=False
        )

        # Determine stable candidates
        stable = []
        for sig in all_signals:
            stepwise_count = sum(
                1 for fr in self.fold_results if sig in fr.stepwise_selected
            )
            score = stepwise_count / self.n_folds
            if (stepwise_count >= self.config.min_fold_survival
                    and score >= self.config.stability_threshold):
                stable.append(sig)

        # Sort stable candidates by stability score (descending)
        stable.sort(key=lambda s: stability_scores.get(s, 0), reverse=True)

        return StabilityReport(
            survival_matrix=survival_matrix,
            stability_scores=stability_scores,
            cross_fold_icir=cross_fold_icir,
            stable_candidates=stable,
            per_signal_summary=per_signal_summary,
        )
