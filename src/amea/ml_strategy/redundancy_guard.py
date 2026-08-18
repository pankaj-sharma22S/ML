"""Redundancy Guard inspecting experiment history to prune duplicate experiments."""

from typing import List, Set
from amea.core.state import RegisteredExperimentRecord
from amea.ml_strategy.models import ExperimentSpecification


class RedundancyGuard:
    """Filters out proposed experiments that duplicate completed runs in the experiment history."""

    @staticmethod
    def filter_redundant_experiments(
        proposed_experiments: List[ExperimentSpecification],
        history: List[RegisteredExperimentRecord],
    ) -> List[ExperimentSpecification]:
        if not history:
            return proposed_experiments

        completed_ids: Set[str] = {r.experiment_id for r in history}
        unique_experiments: List[ExperimentSpecification] = []

        for exp in proposed_experiments:
            if exp.experiment_id in completed_ids:
                continue
            unique_experiments.append(exp)

        return unique_experiments
