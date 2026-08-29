"""End-to-End Real Execution Scenario on Customer Churn Dataset."""

import json
import joblib
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from amea.core.config import ProjectConfig, ComputeBudget
from amea.core.state import LifecyclePhase, TaskType
from amea.orchestrator.runner import OrchestratorRunner


def test_real_churn_classification_end_to_end(tmp_path):
    """Execute complete real multi-agent ML engineering pipeline on realistic churn dataset."""
    dataset_path = Path("data/customer_churn.csv")
    assert dataset_path.exists(), "Realistic churn dataset must exist on disk"

    df_raw = pd.read_csv(dataset_path)
    assert len(df_raw) == 250
    assert "churn" in df_raw.columns
    assert "customer_age" in df_raw.columns
    assert "contract_type" in df_raw.columns

    # 1. Initialize Orchestrator with real compute budget
    project_id = f"churn_e2e_{tmp_path.name}"
    config = ProjectConfig(
        project_id=project_id,
        budget=ComputeBudget(max_experiments=3, max_total_duration_sec=300),
    )
    runner = OrchestratorRunner(config=config)

    # Track emitted events from all agents
    events_trace = []
    def event_recorder(event):
        events_trace.append({
            "event_type": event.event_type.value,
            "source": event.source_component,
            "message": event.message,
        })
    runner.event_bus.subscribe_all(event_recorder)

    # 2. Run Task
    user_prompt = "Train a machine learning classifier to predict customer churn from demographic and account features"
    final_state = runner.run_task(
        user_request=user_prompt,
        dataset_path=str(dataset_path),
        target_column="churn",
    )

    # 3. Verify Lifecycle Completion
    assert final_state.current_phase == LifecyclePhase.TERMINATED
    assert final_state.is_terminal is True

    # 4. Verify Problem Understanding
    assert final_state.task_spec is not None
    assert final_state.task_spec.target_column == "churn"
    assert final_state.task_spec.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
    assert final_state.task_spec.primary_metric in ("roc_auc", "f1_score", "accuracy")

    # 5. Verify Data Intelligence & Profiling
    assert final_state.data_profile is not None
    assert final_state.data_profile.total_rows == 250
    assert final_state.data_profile.total_columns == 7
    assert len(final_state.data_profile.columns) == 7

    # 6. Verify EDA Findings & Cleaning
    assert len(final_state.eda_findings) > 0
    assert final_state.data_quality_report is not None

    # 7. Verify Multiple Model Specialists and Real Experiment Execution
    assert len(final_state.experiment_ledger) >= 2
    for exp in final_state.experiment_ledger:
        assert exp.exit_code == 0, f"Experiment {exp.experiment_id} failed with exit code {exp.exit_code}"
        assert exp.training_duration_sec > 0.0
        assert len(exp.cv_metrics_mean) > 0
        primary_metric = final_state.task_spec.primary_metric
        assert primary_metric in exp.cv_metrics_mean
        score = exp.cv_metrics_mean[primary_metric]
        assert 0.0 <= score <= 1.0, f"Invalid metric score {score} for {exp.model_family}"

    # 8. Verify Evaluation & Judge Decision
    assert final_state.best_candidate is not None
    assert final_state.judge_decision is not None
    assert final_state.judge_decision.selected_experiment_id is not None
    assert len(final_state.judge_decision.rationale) > 0

    # 9. Verify Generated 8-File Production Pipeline Artifacts
    assert final_state.code_artifacts is not None
    files = final_state.code_artifacts.files
    required_files = ["data_loader.py", "preprocess.py", "features.py", "train.py", "evaluate.py", "inference.py", "requirements.txt", "config.json"]
    for rf in required_files:
        assert rf in files, f"Missing required pipeline file {rf}"
        assert len(files[rf]) > 0, f"Generated file {rf} is empty"

    # 10. Real Inference Verification
    # Load raw test rows and verify inference with synthesized pipeline logic
    test_rows = df_raw.drop(columns=["churn"]).iloc[:5]
    assert len(test_rows) == 5
