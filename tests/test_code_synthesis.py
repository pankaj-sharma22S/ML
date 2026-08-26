"""Unit and integration tests for the Code Synthesis Agent."""

import ast
from pathlib import Path
import pytest

from amea.code_synthesis.agent import CodeSynthesisAgent
from amea.code_synthesis.models import CodeSynthesisContext
from amea.code_synthesis.validator import CodeSyntaxValidator
from amea.core.state import (
    DataProfile,
    JudgeDecision,
    MLTaskSpecification,
    RegisteredExperimentRecord,
    TaskType,
)


@pytest.fixture
def mock_synthesis_context(tmp_path):
    task_spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        primary_metric="roc_auc",
        secondary_metrics=["accuracy", "f1"],
    )

    data_profile = DataProfile(
        dataset_path=str(tmp_path / "test_data.csv"),
        dataset_sha256="dummy_hash_123",
        total_rows=200,
        total_columns=6,
        memory_footprint_mb=0.5,
        columns={},
    )

    best_candidate = RegisteredExperimentRecord(
        experiment_id="exp_best_linear",
        model_family="LinearModel",
        hyperparameters={"C": 1.0, "max_iter": 500, "random_state": 42},
        cv_metrics_mean={"roc_auc": 0.92, "accuracy": 0.88, "f1": 0.87},
        cv_metrics_std={"roc_auc": 0.01},
        train_metrics_mean={"roc_auc": 0.95},
        training_duration_sec=1.5,
        inference_latency_ms=1.2,
        peak_memory_mb=64.0,
        exit_code=0,
    )

    judge_decision = JudgeDecision(
        action="ACCEPT_BEST_CANDIDATE",
        selected_experiment_id="exp_best_linear",
        rationale="Selected LinearModel with verified audit",
    )

    return CodeSynthesisContext(
        task_spec=task_spec,
        data_profile=data_profile,
        best_candidate=best_candidate,
        judge_decision=judge_decision,
    )


def test_code_synthesis_agent_generates_all_files(tmp_path, mock_synthesis_context):
    agent = CodeSynthesisAgent(base_output_dir=tmp_path / "generated")
    artifacts = agent.synthesize(mock_synthesis_context)

    expected_files = [
        "data_loader.py",
        "preprocess.py",
        "features.py",
        "train.py",
        "evaluate.py",
        "inference.py",
        "requirements.txt",
        "config.json",
    ]

    for fname in expected_files:
        assert fname in artifacts.files
        file_path = Path(artifacts.pipeline_dir) / fname
        assert file_path.exists()
        assert len(file_path.read_text(encoding="utf-8")) > 0

    assert artifacts.validation_report.is_valid_syntax is True
    assert len(artifacts.validation_report.syntax_errors) == 0


def test_code_syntax_validator_ast_checks(mock_synthesis_context):
    agent = CodeSynthesisAgent()
    artifacts = agent.synthesize(mock_synthesis_context)

    # Verify that all Python files parse cleanly via ast.parse
    for fname, code in artifacts.files.items():
        if fname.endswith(".py"):
            parsed_tree = ast.parse(code)
            assert parsed_tree is not None

    # Test error detection
    corrupted_files = dict(artifacts.files)
    corrupted_files["train.py"] = "def invalid_python_syntax(:\n  return"
    report = CodeSyntaxValidator.validate_pipeline_code(corrupted_files, mock_synthesis_context)
    assert report.is_valid_syntax is False
    assert "train.py" in report.syntax_errors


def test_code_synthesis_tree_and_boosting_models(tmp_path):
    task_spec = MLTaskSpecification(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        primary_metric="accuracy",
    )
    best_rf = RegisteredExperimentRecord(
        experiment_id="exp_rf_1",
        model_family="RandomForest",
        hyperparameters={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        cv_metrics_mean={"accuracy": 0.89},
        cv_metrics_std={"accuracy": 0.02},
        train_metrics_mean={"accuracy": 0.94},
        training_duration_sec=2.0,
        inference_latency_ms=3.0,
        peak_memory_mb=120.0,
        exit_code=0,
    )

    context = CodeSynthesisContext(
        task_spec=task_spec,
        best_candidate=best_rf,
    )

    agent = CodeSynthesisAgent(base_output_dir=tmp_path / "generated")
    artifacts = agent.synthesize(context)

    assert "RandomForestClassifier" in artifacts.files["train.py"]
    assert artifacts.validation_report.is_valid_syntax is True
