"""End-to-end integration test running full ML pipeline from ingestion to verified delivery."""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from amea.core.config import ProjectConfig, ComputeBudget
from amea.core.state import LifecyclePhase
from amea.orchestrator.runner import OrchestratorRunner


def test_orchestrator_end_to_end_run(tmp_path):
    # 1. Create a real synthetic classification dataset
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    df["target"] = y

    data_file = tmp_path / "sample_data.csv"
    df.to_csv(data_file, index=False)

    # 2. Configure runner with temporary sandbox and persistence
    config = ProjectConfig(
        project_id="e2e_test_proj",
        persistence={"project_dir": tmp_path / "persistence"},
        security={"sandbox_root": tmp_path / "sandboxes"},
        budget=ComputeBudget(max_experiments=5),
    )

    runner = OrchestratorRunner(config=config)

    # 3. Execute Task
    final_state = runner.run_task(
        user_request="Train a classification model to predict target from features",
        dataset_path=str(data_file),
        target_column="target",
    )

    # 4. Verify Assertions
    assert final_state.is_terminal
    assert final_state.data_profile is not None
    assert final_state.data_profile.total_rows == 100
    assert final_state.data_profile.total_columns == 5

    # Verify experiments ran
    assert len(final_state.experiment_ledger) > 0
    for exp in final_state.experiment_ledger:
        assert exp.exit_code == 0
        assert "roc_auc" in exp.cv_metrics_mean

    # Verify best candidate selected
    assert final_state.best_candidate is not None
    assert final_state.best_candidate.model_family in ("RandomForest", "LinearModel")

    # Verify code artifacts generated
    assert final_state.code_artifacts is not None
    assert "train.py" in final_state.code_artifacts.files
    assert "data_loader.py" in final_state.code_artifacts.files

    # Verify sandbox execution of generated code succeeded
    assert final_state.execution_result is not None
    assert final_state.execution_result.is_verified
    assert final_state.execution_result.exit_code == 0

    # Verify final report
    assert final_state.final_report is not None
    assert final_state.final_report.total_experiments_run == len(final_state.experiment_ledger)
