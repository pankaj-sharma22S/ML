"""Experiment Planner sequencing model candidates into DAGs and parallel-safe groups."""

from typing import List
from amea.ml_strategy.models import (
    ExperimentBudgetSpec,
    ExperimentSpecification,
    FeatureEngineeringHypothesis,
    ModelCandidate,
)


class ExperimentPlanner:
    """Sequences model candidates into ordered, parallel-safe experiment specifications."""

    @staticmethod
    def plan_experiments(
        candidates: List[ModelCandidate],
        hypotheses: List[FeatureEngineeringHypothesis],
        budget: ExperimentBudgetSpec,
        random_seed: int = 42,
    ) -> List[ExperimentSpecification]:
        experiments: List[ExperimentSpecification] = []

        # Map hypotheses to preprocessing names
        preproc_steps = [h.transformation_name for h in hypotheses[:2]]
        if not preproc_steps:
            preproc_steps = ["SimpleImputer", "StandardScaler"]

        for i, cand in enumerate(candidates):
            if len(experiments) >= budget.max_experiments:
                break

            exp_id = f"exp_{cand.candidate_id}_seed{random_seed}"
            experiments.append(
                ExperimentSpecification(
                    experiment_id=exp_id,
                    hypothesis=f"Evaluate {cand.model_family.value} ({cand.model_class_name}) with baseline preprocessing.",
                    model_family=cand.model_family,
                    model_class_name=cand.model_class_name,
                    preprocessing_steps=preproc_steps,
                    hyperparameters=cand.default_hyperparameters,
                    seed=random_seed,
                    parallel_group_id="parallel_group_1",
                    dependencies=[],
                    timeout_seconds=budget.timeout_per_experiment_sec,
                    priority=cand.priority,
                )
            )

        return experiments
