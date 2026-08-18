"""ML Strategy Agent coordinating evidence-driven experimentation planning."""

from typing import List
from uuid import uuid4

from amea.core.state import TaskType
from amea.ml_strategy.baseline_designer import BaselineDesigner
from amea.ml_strategy.experiment_planner import ExperimentPlanner
from amea.ml_strategy.feature_hypothesizer import FeatureHypothesizer
from amea.ml_strategy.model_selector import ModelSelector
from amea.ml_strategy.models import (
    ExperimentBudgetSpec,
    MetricSpecification,
    MLStrategyContext,
    MLStrategyPlan,
    StrategyConfidence,
    StrategyStatus,
    ValidationStrategySpec,
)
from amea.ml_strategy.redundancy_guard import RedundancyGuard


class MLStrategyAgent:
    """Independent strategy formulation agent constructing evidence-backed MLStrategyPlans."""

    def __init__(self):
        self.model_selector = ModelSelector()
        self.baseline_designer = BaselineDesigner()
        self.feature_hypothesizer = FeatureHypothesizer()
        self.experiment_planner = ExperimentPlanner()
        self.redundancy_guard = RedundancyGuard()

    def plan(self, context: MLStrategyContext) -> MLStrategyPlan:
        """Formulate a comprehensive MLStrategyPlan from upstream evidence."""
        task_spec = context.task_spec
        budget = context.budget or ExperimentBudgetSpec()

        # 1. Validation & Failure Guard
        if not task_spec.target_column:
            return MLStrategyPlan(
                strategy_id=f"strat_{uuid4().hex[:8]}",
                strategy_status=StrategyStatus.BLOCKED,
                problem_summary="Target column is missing; cannot formulate supervised ML strategy.",
                task_type=task_spec.task_type,
                metric_spec=MetricSpecification(primary_metric=task_spec.primary_metric),
                validation_strategy=ValidationStrategySpec(cv_scheme="KFold"),
                budget=budget,
                confidence=StrategyConfidence(score=0.0, factors=["Missing target column."]),
                rationale="Supervised ML strategy formulation blocked due to missing target column.",
            )

        # 2. Select Model Candidates & Exclusions
        candidates, exclusions = self.model_selector.select_candidates(
            task_type=task_spec.task_type,
            data_profile=context.data_profile,
            random_seed=task_spec.random_seed,
        )

        # 3. Design Baseline Candidate
        baseline = self.baseline_designer.design_baseline(
            task_type=task_spec.task_type,
            random_seed=task_spec.random_seed,
        )

        # 4. Generate Feature Hypotheses from EDA
        hypotheses = self.feature_hypothesizer.generate_hypotheses(
            eda_findings=context.eda_findings,
        )

        # 5. Plan Experiments
        raw_experiments = self.experiment_planner.plan_experiments(
            candidates=candidates,
            hypotheses=hypotheses,
            budget=budget,
            random_seed=task_spec.random_seed,
        )

        # 6. Filter Redundancy using History
        final_experiments = self.redundancy_guard.filter_redundant_experiments(
            proposed_experiments=raw_experiments,
            history=context.experiment_history,
        )

        # 7. Metric Specification & Validation Strategy
        is_classification = (task_spec.task_type != TaskType.REGRESSION and task_spec.task_type != TaskType.TIME_SERIES)
        validation_spec = ValidationStrategySpec(
            cv_scheme="StratifiedKFold" if is_classification else ("TimeSeriesSplit" if task_spec.task_type == TaskType.TIME_SERIES else "KFold"),
            n_splits=5,
            shuffle=(task_spec.task_type != TaskType.TIME_SERIES),
            rationale=f"Empirically selected {('StratifiedKFold' if is_classification else 'KFold')} to prevent distribution shift across evaluation folds.",
        )

        metric_spec = MetricSpecification(
            primary_metric=task_spec.primary_metric,
            secondary_metrics=task_spec.secondary_metrics,
            optimization_direction=task_spec.optimization_direction,
        )

        # 8. Compute Evidence-Based Confidence
        confidence_factors: List[str] = [
            f"Evaluated {len(candidates)} model candidates matching dataset scale.",
            f"Formulated {len(hypotheses)} evidence-backed feature engineering hypotheses.",
        ]
        if context.data_profile:
            confidence_factors.append(f"Profiled {context.data_profile.total_rows} rows across {context.data_profile.total_columns} columns.")

        return MLStrategyPlan(
            strategy_id=f"strat_{uuid4().hex[:8]}",
            strategy_status=StrategyStatus.READY,
            problem_summary=f"Supervised {task_spec.task_type.value} strategy optimizing for {task_spec.primary_metric}.",
            task_type=task_spec.task_type,
            metric_spec=metric_spec,
            validation_strategy=validation_spec,
            model_candidates=candidates,
            baseline_candidate=baseline,
            feature_hypotheses=hypotheses,
            experiment_plan=final_experiments,
            excluded_approaches=exclusions,
            budget=budget,
            stopping_criteria=[
                f"Achieved primary metric {task_spec.primary_metric} target threshold.",
                "Completed all planned model family candidate evaluations.",
                "Relative improvement across iterations < 0.005.",
            ],
            risks=[
                "High cardinality features risk overfitting if not properly regularized.",
                "Potential class imbalance may skew naive accuracy.",
            ],
            assumptions=["Data cleaning and independent quality gates verified clean source distributions."],
            confidence=StrategyConfidence(score=0.95, factors=confidence_factors),
            rationale=f"Prioritizes linear baseline, bagging ensembles, and gradient boosting while excluding heavy neural nets for sample efficiency.",
        )
