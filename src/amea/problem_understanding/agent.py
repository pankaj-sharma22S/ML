"""Problem Understanding Agent coordinating objective formulation and conflict reconciliation."""

from typing import Dict, List, Optional
import pandas as pd

from amea.core.state import MLTaskSpecification
from amea.problem_understanding.intent_parser import IntentParser
from amea.problem_understanding.models import ProblemUnderstandingReport
from amea.problem_understanding.task_arbitrator import TaskArbitrator
from amea.problem_understanding.metric_recommender import MetricRecommender
from amea.problem_understanding.validation_recommender import ValidationRecommender


class ProblemUnderstandingAgent:
    """Independent agent formalizing user requests and dataset evidence into an authoritative MLTaskSpecification."""

    def __init__(self):
        self.intent_parser = IntentParser()
        self.task_arbitrator = TaskArbitrator()
        self.metric_recommender = MetricRecommender()
        self.validation_recommender = ValidationRecommender()

    def formulate_problem(
        self,
        user_request: str,
        df: Optional[pd.DataFrame] = None,
        target_column_hint: Optional[str] = None,
        random_seed: int = 42,
    ) -> ProblemUnderstandingReport:
        """Parse intent, reconcile with empirical data evidence, and build ProblemUnderstandingReport."""
        # 1. Parse natural language intent
        intent = self.intent_parser.parse(user_request)

        # 2. Arbitrate TaskType and Target Column
        task_type, target, conflicts, assumptions = self.task_arbitrator.arbitrate(
            intent=intent,
            df=df,
            target_column=target_column_hint,
        )

        # 3. Recommend Metrics & Optimization Direction
        primary_metric, secondary_metrics, direction = self.metric_recommender.recommend(
            task_type=task_type,
            intent=intent,
            df=df,
            target_column=target,
        )

        # 4. Recommend Validation Strategy
        cv_strategy = self.validation_recommender.recommend(
            task_type=task_type,
            df=df,
            target_column=target,
        )

        # 5. Extract feature candidates
        feature_candidates = []
        if df is not None:
            feature_candidates = [c for c in df.columns if c != target]

        spec = MLTaskSpecification(
            task_type=task_type,
            target_column=target,
            feature_candidates=feature_candidates,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            optimization_direction=direction,
            random_seed=random_seed,
        )

        blocking_conflicts = [c for c in conflicts if c.severity == "BLOCKING"]
        is_feasible = (len(blocking_conflicts) == 0)

        return ProblemUnderstandingReport(
            task_spec=spec,
            intent_analysis=intent,
            conflicts=conflicts,
            identified_gaps=[c.description for c in blocking_conflicts],
            assumptions_made=assumptions,
            validation_strategy=cv_strategy,
            is_feasible=is_feasible,
            blocking_reason=blocking_conflicts[0].description if blocking_conflicts else None,
        )
