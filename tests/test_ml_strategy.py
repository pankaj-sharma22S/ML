"""Unit and integration tests for the ML Strategy Agent."""

from pathlib import Path
import pytest

from amea.core.state import (
    ColumnProfile,
    DataProfile,
    MLTaskSpecification,
    RegisteredExperimentRecord,
    TaskType,
)
from amea.ml_strategy.agent import MLStrategyAgent
from amea.ml_strategy.baseline_designer import BaselineDesigner
from amea.ml_strategy.feature_hypothesizer import FeatureHypothesizer
from amea.ml_strategy.model_selector import ModelSelector
from amea.ml_strategy.models import (
    ExperimentBudgetSpec,
    MLStrategyContext,
    ModelFamily,
    StrategyStatus,
)
from amea.ml_strategy.redundancy_guard import RedundancyGuard


def test_model_selector_small_dataset():
    # Small dataset (< 30 rows) -> RF & GB excluded, only linear
    profile = DataProfile(
        dataset_path="small.csv",
        dataset_sha256="hash_small_123",
        total_rows=20,
        total_columns=5,
        memory_footprint_mb=0.1,
        columns={},
    )
    candidates, exclusions = ModelSelector.select_candidates(
        task_type=TaskType.BINARY_CLASSIFICATION,
        data_profile=profile,
    )
    assert len(candidates) == 1
    assert candidates[0].model_family == ModelFamily.LINEAR_MODEL
    assert any(e.family_or_technique == "RandomForest" for e in exclusions)
    assert any(e.family_or_technique == "GradientBoosting" for e in exclusions)


def test_model_selector_medium_dataset():
    # Medium dataset (500 rows) -> Linear, RF, GB included; TabularNeuralNet excluded
    profile = DataProfile(
        dataset_path="medium.csv",
        dataset_sha256="hash_medium_123",
        total_rows=500,
        total_columns=10,
        memory_footprint_mb=1.0,
        columns={},
    )
    candidates, exclusions = ModelSelector.select_candidates(
        task_type=TaskType.BINARY_CLASSIFICATION,
        data_profile=profile,
    )
    assert len(candidates) == 3
    candidate_families = [c.model_family for c in candidates]
    assert ModelFamily.LINEAR_MODEL in candidate_families
    assert ModelFamily.RANDOM_FOREST in candidate_families
    assert ModelFamily.GRADIENT_BOOSTING in candidate_families
    assert any(e.family_or_technique == "TabularNeuralNet" for e in exclusions)


def test_baseline_designer():
    base_clf = BaselineDesigner.design_baseline(TaskType.BINARY_CLASSIFICATION)
    assert base_clf.model_family == ModelFamily.LINEAR_MODEL
    assert base_clf.model_class_name == "LogisticRegression"

    base_reg = BaselineDesigner.design_baseline(TaskType.REGRESSION)
    assert base_reg.model_family == ModelFamily.LINEAR_MODEL
    assert base_reg.model_class_name == "Ridge"


def test_feature_hypothesizer():
    findings = [
        "[IMPORTANT] Feature 'income' is strongly right-skewed (skewness = 3.5).",
        "[IMPORTANT] Categorical feature 'city' has high cardinality (85 distinct values).",
    ]
    hypotheses = FeatureHypothesizer.generate_hypotheses(findings)
    assert len(hypotheses) == 2
    assert any("PowerTransformer" in h.transformation_name for h in hypotheses)
    assert any("TargetEncoding" in h.transformation_name for h in hypotheses)


def test_redundancy_guard():
    agent = MLStrategyAgent()
    spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        primary_metric="roc_auc",
    )
    profile = DataProfile(
        dataset_path="data.csv",
        dataset_sha256="hash_data_123",
        total_rows=300,
        total_columns=5,
        memory_footprint_mb=0.5,
        columns={},
    )
    context = MLStrategyContext(task_spec=spec, data_profile=profile)
    plan1 = agent.plan(context)
    assert len(plan1.experiment_plan) >= 2

    # Simulate completed experiment
    completed = [
        RegisteredExperimentRecord(
            experiment_id=plan1.experiment_plan[0].experiment_id,
            model_family="LinearModel",
            hyperparameters={},
            cv_metrics_mean={"roc_auc": 0.85},
            cv_metrics_std={"roc_auc": 0.02},
            train_metrics_mean={"roc_auc": 0.88},
            training_duration_sec=1.2,
            inference_latency_ms=1.5,
            peak_memory_mb=128.0,
            exit_code=0,
        )
    ]

    filtered = RedundancyGuard.filter_redundant_experiments(plan1.experiment_plan, completed)
    assert len(filtered) == len(plan1.experiment_plan) - 1
    assert plan1.experiment_plan[0].experiment_id not in [e.experiment_id for e in filtered]


def test_ml_strategy_agent_blocked_on_missing_target():
    agent = MLStrategyAgent()
    spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column=None,  # Missing target!
        primary_metric="roc_auc",
    )
    context = MLStrategyContext(task_spec=spec)
    plan = agent.plan(context)
    assert plan.strategy_status == StrategyStatus.BLOCKED
    assert plan.confidence.score == 0.0


def test_ml_strategy_agent_end_to_end():
    agent = MLStrategyAgent()
    spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        primary_metric="roc_auc",
        secondary_metrics=["f1", "accuracy"],
    )
    profile = DataProfile(
        dataset_path="churn.csv",
        dataset_sha256="hash_churn_123",
        total_rows=1000,
        total_columns=10,
        memory_footprint_mb=2.0,
        columns={},
    )
    context = MLStrategyContext(
        task_spec=spec,
        data_profile=profile,
        eda_findings=["[IMPORTANT] Feature 'tenure' is right-skewed."],
        budget=ExperimentBudgetSpec(max_experiments=4),
    )

    plan = agent.plan(context)
    assert plan.strategy_status == StrategyStatus.READY
    assert plan.confidence.score >= 0.90
    assert len(plan.model_candidates) == 3
    assert len(plan.experiment_plan) <= 4
    assert plan.validation_strategy.cv_scheme == "StratifiedKFold"
    assert len(plan.feature_hypotheses) >= 1
