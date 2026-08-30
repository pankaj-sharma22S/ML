"""Comprehensive Smoke and Integration Tests for Real AMEA ML Workspace."""

import os
import sys
import json
import uuid
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from amea.server import app
from amea.llm.base import ProviderType
from amea.llm.factory import LLMProviderFactory
from amea.llm.openrouter import OpenRouterProvider
from amea.llm.ollama import OllamaProvider
from amea.core.env import mask_secret, scrub_secrets_from_text
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.kernel.execution_request import ExecuteCellRequest

client = TestClient(app)


# ============================================================
# 1. Real Demo Data & Schema Inspection Tests
# ============================================================

def test_real_churn_dataset_exists_and_valid():
    """Verify real churn CSV is present with realistic schema and valid values."""
    csv_path = Path("data/customer_churn.csv")
    assert csv_path.exists(), "data/customer_churn.csv must exist"
    
    df = pd.read_csv(csv_path)
    assert len(df) >= 200
    expected_cols = {"customer_age", "tenure_months", "monthly_charges", "contract_type", "support_calls", "payment_method", "churn"}
    assert expected_cols.issubset(set(df.columns))
    assert df["churn"].nunique() == 2
    assert set(df["churn"].unique()).issubset({0, 1})


def test_dataset_upload_and_automatic_profiling(tmp_path):
    """Verify CSV dataset upload and instant schema profiling."""
    test_csv = tmp_path / "upload_test.csv"
    test_df = pd.DataFrame({
        "age": [25, 45, 65, 30],
        "income": [50000.0, 80000.0, np.nan, 45000.0],
        "category": ["A", "B", "A", "C"],
        "default": [0, 1, 0, 0]
    })
    test_df.to_csv(test_csv, index=False)

    with open(test_csv, "rb") as f:
        res = client.post(
            "/api/project/upload-dataset",
            files={"file": ("upload_test.csv", f, "text/csv")},
            data={"project_path": str(tmp_path)},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "uploaded"
    assert data["total_rows"] == 4
    assert data["total_columns"] == 4
    assert "default" in data["candidate_targets"]
    assert len(data["columns"]) == 4
    assert len(data["preview_records"]) == 4


# ============================================================
# 2. Interactive Notebook & Kernel Variable Persistence
# ============================================================

def test_kernel_cell_state_persistence_and_tracebacks(tmp_path):
    """Verify real Python cell execution with shared state and exception tracebacks."""
    from amea.execution.kernel.kernel_executor import KernelExecutor
    km = KernelManager()
    executor = KernelExecutor(km)
    session = km.create_session("notebook_test_proj")

    try:
        # 1. Cell 1: Define variables and compute
        req1 = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="c1",
            code="import numpy as np\nx = 100\nvec = np.array([10, 20, 30])\nprint(f'Computed sum: {vec.sum()}')",
        )
        res1 = executor.execute_cell(req1)
        assert res1.is_success is True
        assert any("Computed sum: 60" in str(getattr(o, "text", "")) for o in res1.outputs)

        # 2. Cell 2: Consume variables from Cell 1
        req2 = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="c2",
            code="y = x * 2 + int(vec.mean())\ny",
        )
        res2 = executor.execute_cell(req2)
        assert res2.is_success is True

        # 3. Cell 3: Intentional Exception & Traceback
        req3 = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="c3",
            code="raise ValueError('Intentional verification error in ML code')",
        )
        res3 = executor.execute_cell(req3)
        assert res3.is_success is False
        assert res3.status == "ERROR"
        assert res3.failure_diagnosis is not None
        error_output = next((o for o in res3.outputs if o.output_type.value == "ERROR"), None)
        assert error_output is not None
        assert error_output.error_name == "ValueError"
        assert "Intentional verification error" in (error_output.error_value or "")
    finally:
        km.shutdown(session.session_id)


# ============================================================
# 3. Terminal Execution, Security & Secret Scrubbing
# ============================================================

def test_terminal_exec_valid_commands():
    """Verify real terminal command execution."""
    res = client.post("/api/terminal/exec", json={"project_path": ".", "command": "python --version"})
    assert res.status_code == 200
    data = res.json()
    assert "Python 3." in (data["stdout"] + data["stderr"])
    assert data["exit_code"] == 0


def test_terminal_security_blocks_destructive_commands():
    """Verify dangerous destructive commands are blocked by security policy."""
    res = client.post("/api/terminal/exec", json={"project_path": ".", "command": "rm -rf /"})
    assert res.status_code == 200
    data = res.json()
    assert data["exit_code"] == 126
    assert data["audit_status"] == "BLOCKED"
    assert "Security Violation" in data["stderr"]


def test_terminal_security_blocks_secret_dumping():
    """Verify reading .env via terminal is blocked."""
    res = client.post("/api/terminal/exec", json={"project_path": ".", "command": "type .env"})
    assert res.status_code == 200
    data = res.json()
    assert data["exit_code"] == 126
    assert data["audit_status"] == "BLOCKED"


def test_secret_scrubber_redacts_tokens():
    """Verify secret scrubbing utility prevents token leakage."""
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-secretkey999988887777"
    raw_output = "Error occurred with sk-or-v1-secretkey999988887777 during connection."
    clean = scrub_secrets_from_text(raw_output)
    assert "sk-or-v1-secretkey999988887777" not in clean
    assert "[REDACTED_SECRET_OPENROUTER_API_KEY]" in clean
    
    assert mask_secret("sk-or-v1-1234567890abcdef") == "sk-o...cdef"


# ============================================================
# 4. LLM Provider Abstraction & Status
# ============================================================

def test_llm_provider_abstraction_and_health_checks():
    """Verify OpenRouter and Ollama provider instantiation and health report."""
    # 1. OpenRouter Provider
    or_provider = OpenRouterProvider(api_key="sk-or-v1-mock-key-12345")
    assert or_provider.has_api_key is True
    assert or_provider.masked_key().startswith("sk-o")

    # 2. Ollama Provider
    ollama_provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
    status_ollama = ollama_provider.health_check()
    assert status_ollama.provider_type == ProviderType.OLLAMA

    # 3. LLM API Status Endpoint
    res = client.get("/api/llm/status")
    assert res.status_code == 200
    data = res.json()
    assert "active_provider" in data
    assert "providers" in data
    assert "openrouter" in data["providers"]
    assert "ollama" in data["providers"]


# ============================================================
# 5. Parallel Multi-Model Real ML Execution & Prediction
# ============================================================

def test_parallel_multi_model_execution_and_prediction(tmp_path):
    """Verify parallel model execution via ExperimentRunner and final prediction."""
    from amea.core.config import ProjectConfig, ComputeBudget
    from amea.orchestrator.runner import OrchestratorRunner
    import joblib

    config = ProjectConfig(
        project_id=f"test_par_{tmp_path.name}",
        budget=ComputeBudget(max_experiments=3, max_total_duration_sec=120),
    )
    runner = OrchestratorRunner(config=config)

    final_state = runner.run_task(
        user_request="Predict customer churn using tabular models",
        dataset_path="data/customer_churn.csv",
        target_column="churn",
    )

    assert final_state.is_terminal is True
    assert len(final_state.experiment_ledger) >= 2
    for exp in final_state.experiment_ledger:
        assert exp.exit_code == 0
        assert "roc_auc" in exp.cv_metrics_mean

    # Winner selected
    assert final_state.best_candidate is not None
    assert final_state.judge_decision is not None

    # Pipeline synthesized
    assert final_state.code_artifacts is not None
    assert "train.py" in final_state.code_artifacts.files
    assert "inference.py" in final_state.code_artifacts.files
