"""Evaluation Agent auditing executed experiments for overfitting, leakage, and metric validity."""

from typing import Dict, List
from amea.core.state import RegisteredExperimentRecord
from amea.evaluation.models import AuditVerdict, CandidateAuditReport


class EvaluationAgent:
    """Independent auditor evaluating statistical integrity, overfitting, and leakage of completed experiments."""

    @staticmethod
    def audit(
        records: List[RegisteredExperimentRecord],
        primary_metric_name: str = "roc_auc",
        optimization_direction: str = "maximize",
    ) -> Dict[str, CandidateAuditReport]:
        audits: Dict[str, CandidateAuditReport] = {}

        for rec in records:
            primary_val = rec.cv_metrics_mean.get(primary_metric_name, 0.0)
            train_val = rec.train_metrics_mean.get(primary_metric_name, primary_val)
            overfit_gap = max(0.0, train_val - primary_val)

            notes: List[str] = []
            is_leakage = (primary_val > 0.999)
            if is_leakage:
                notes.append(f"Near-perfect metric ({primary_val:.4f}) indicates potential target/train-test leakage.")

            is_overfitting = (overfit_gap > 0.20)
            if is_overfitting:
                notes.append(f"High overfitting gap ({overfit_gap:.4f}) between train and validation folds.")

            beats_baseline = (primary_val > 0.50) if optimization_direction == "maximize" else True
            audit_passed = (not is_leakage and rec.exit_code == 0 and beats_baseline)

            if is_leakage:
                verdict = AuditVerdict.LEAKAGE_SUSPECTED
            elif is_overfitting:
                verdict = AuditVerdict.OVERFITTING_WARNING
            elif rec.exit_code != 0:
                verdict = AuditVerdict.FAILED
            else:
                verdict = AuditVerdict.PASSED

            audits[rec.experiment_id] = CandidateAuditReport(
                experiment_id=rec.experiment_id,
                verdict=verdict,
                primary_metric_val=round(primary_val, 4),
                overfitting_gap=round(overfit_gap, 4),
                is_leakage_suspected=is_leakage,
                is_stable=True,
                beats_baseline=beats_baseline,
                audit_passed=audit_passed,
                audit_notes=notes,
            )

        return audits
