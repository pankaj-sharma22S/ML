"""Unit and integration tests for Model Specialist Agents, Experiment Runner, Evaluation, and Judge Agents."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from amea.core.state import MLTaskSpecification, RegisteredExperimentRecord, TaskType
from amea.evaluation.auditor import EvaluationAgent
from amea.evaluation.judge import JudgeAgent
from amea.evaluation.models import AuditVerdict
from amea.experiments.models import ExperimentStatus
from amea.experiments.runner import ExperimentRunner
from amea.ml_strategy.models import ExperimentSpecification, ModelFamily
from amea.model_specialists.boosting_specialist import BoostingSpecialistAgent
from amea.model_specialists.linear_specialist import LinearSpecialistAgent
from amea.model_specialists.neural_specialist import NeuralSpecialistAgent
from amea.model_specialists.registry import ModelSpecialistRegistry
from amea.model_specialists.tree_specialist import TreeModelSpecialistAgent


def test_specialist_capability_declarations():
    linear_agent = LinearSpecialistAgent()
    tree_agent = TreeModelSpecialistAgent()
    boosting_agent = BoostingSpecialistAgent()
    neural_agent = NeuralSpecialistAgent()

    assert linear_agent.capabilities().model_family == ModelFamily.LINEAR_MODEL
    assert linear_agent.capabilities().requires_scaling is True

    assert tree_agent.capabilities().model_family == ModelFamily.RANDOM_FOREST
    assert tree_agent.capabilities().requires_scaling is False

    assert boosting_agent.capabilities().model_family == ModelFamily.GRADIENT_BOOSTING
    assert boosting_agent.capabilities().handles_missing_values_natively is True

    assert neural_agent.capabilities().model_family == ModelFamily.TABULAR_NEURAL_NET
    assert neural_agent.capabilities().requires_scaling is True


def test_specialist_registry_lookup():
    registry = ModelSpecialistRegistry()
    assert isinstance(registry.get_specialist(ModelFamily.LINEAR_MODEL), LinearSpecialistAgent)
    assert isinstance(registry.get_specialist("RandomForest"), TreeModelSpecialistAgent)
    assert isinstance(registry.get_specialist("GradientBoosting"), BoostingSpecialistAgent)
    assert isinstance(registry.get_specialist("TabularNeuralNet"), NeuralSpecialistAgent)
    assert len(registry.list_capabilities()) == 4


def test_specialist_prepare_execution_and_runner(tmp_path):
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_1": np.random.randn(100),
        "feat_2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100),
    })
    data_path = tmp_path / "sample_data.csv"
    df.to_csv(data_path, index=False)

    task_spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        primary_metric="roc_auc",
    )

    exp_spec = ExperimentSpecification(
        experiment_id="test_exp_linear",
        hypothesis="Test linear logistic regression",
        model_family=ModelFamily.LINEAR_MODEL,
        model_class_name="LogisticRegression",
        hyperparameters={"C": 1.0, "max_iter": 200},
        seed=42,
    )

    linear_agent = LinearSpecialistAgent()
    val_res = linear_agent.validate_experiment(exp_spec, task_spec)
    assert val_res.is_compatible

    exec_config = linear_agent.prepare_execution(exp_spec, task_spec, dataset_path=str(data_path))
    assert exec_config.model_class_name == "LogisticRegression"

    # Execute via ExperimentRunner
    runner = ExperimentRunner(base_workspace_dir=tmp_path / "sandboxes")
    res = runner.run_experiment(exec_config)

    assert res.status == ExperimentStatus.SUCCESS
    assert res.exit_code == 0
    assert "roc_auc" in res.cv_metrics_mean
    assert res.cv_metrics_mean["roc_auc"] > 0.0
    assert Path(res.workspace_dir).exists()


def test_evaluation_agent_and_judge():
    records = [
        RegisteredExperimentRecord(
            experiment_id="exp_normal",
            model_family="LinearModel",
            hyperparameters={},
            cv_metrics_mean={"roc_auc": 0.85},
            cv_metrics_std={"roc_auc": 0.02},
            train_metrics_mean={"roc_auc": 0.88},
            training_duration_sec=1.5,
            inference_latency_ms=1.2,
            peak_memory_mb=64.0,
            exit_code=0,
        ),
        RegisteredExperimentRecord(
            experiment_id="exp_leakage",
            model_family="RandomForest",
            hyperparameters={},
            cv_metrics_mean={"roc_auc": 1.0},  # Suspicious near-perfect metric
            cv_metrics_std={"roc_auc": 0.0},
            train_metrics_mean={"roc_auc": 1.0},
            training_duration_sec=2.5,
            inference_latency_ms=3.0,
            peak_memory_mb=128.0,
            exit_code=0,
        ),
        RegisteredExperimentRecord(
            experiment_id="exp_overfit",
            model_family="GradientBoosting",
            hyperparameters={},
            cv_metrics_mean={"roc_auc": 0.60},
            cv_metrics_std={"roc_auc": 0.05},
            train_metrics_mean={"roc_auc": 0.95},  # Gap = 0.35 > 0.20
            training_duration_sec=3.0,
            inference_latency_ms=2.0,
            peak_memory_mb=90.0,
            exit_code=0,
        ),
    ]

    # Evaluation Agent Audit
    audits = EvaluationAgent.audit(records, primary_metric_name="roc_auc")
    assert audits["exp_normal"].verdict == AuditVerdict.PASSED
    assert audits["exp_normal"].audit_passed is True

    assert audits["exp_leakage"].verdict == AuditVerdict.LEAKAGE_SUSPECTED
    assert audits["exp_leakage"].audit_passed is False

    assert audits["exp_overfit"].verdict == AuditVerdict.OVERFITTING_WARNING

    # Judge Agent Decision
    best, decision = JudgeAgent.evaluate(records, audits, primary_metric_name="roc_auc")
    assert best.experiment_id == "exp_normal"  # Only valid candidate that passed audit
    assert decision.selected_experiment_id == "exp_normal"
    assert len(decision.pareto_rankings) >= 1
