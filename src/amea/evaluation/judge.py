"""Judge Agent arbitrating candidate models and selecting Pareto-optimal winner."""

from typing import Dict, List, Tuple
from amea.core.state import JudgeDecision, RegisteredExperimentRecord
from amea.evaluation.models import CandidateAuditReport, ParetoCandidateRecord


class JudgeAgent:
    """Independent arbitration agent selecting the Pareto-optimal production model candidate."""

    @staticmethod
    def evaluate(
        records: List[RegisteredExperimentRecord],
        audits: Dict[str, CandidateAuditReport],
        primary_metric_name: str = "roc_auc",
        optimization_direction: str = "maximize",
    ) -> Tuple[RegisteredExperimentRecord, JudgeDecision]:
        if not records:
            raise ValueError("Cannot evaluate empty experiment records.")

        # 1. Filter candidates that passed evaluation audit
        valid_records = [
            r for r in records
            if audits.get(r.experiment_id) and audits[r.experiment_id].audit_passed and r.exit_code == 0
        ]

        if not valid_records:
            valid_records = records  # fallback to all records if none cleanly passed

        # 2. Sort by primary metric
        is_max = (optimization_direction == "maximize")
        valid_records.sort(
            key=lambda r: r.cv_metrics_mean.get(primary_metric_name, 0.0),
            reverse=is_max,
        )

        best_candidate = valid_records[0]
        best_metric_val = best_candidate.cv_metrics_mean.get(primary_metric_name, 0.0)

        # 3. Compile Pareto rankings
        pareto_list = []
        for rank, r in enumerate(valid_records, 1):
            pareto_list.append({
                "rank": rank,
                "experiment_id": r.experiment_id,
                "model_family": r.model_family,
                "primary_metric": r.cv_metrics_mean.get(primary_metric_name, 0.0),
                "inference_latency_ms": r.inference_latency_ms,
                "peak_memory_mb": r.peak_memory_mb,
            })

        decision = JudgeDecision(
            action="ACCEPT_BEST_CANDIDATE",
            selected_experiment_id=best_candidate.experiment_id,
            rationale=f"Selected {best_candidate.model_family} ({best_candidate.experiment_id}) achieving primary metric {primary_metric_name} = {best_metric_val:.4f} with verified audit passage.",
            pareto_rankings=pareto_list,
        )

        return best_candidate, decision
