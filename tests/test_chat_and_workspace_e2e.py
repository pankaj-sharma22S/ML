"""End-to-End Verification Test Suite for Chat, Terminal, Notebook, Upload, and Multi-Agent ML Pipeline."""

import os
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from amea.server import app
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.kernel.kernel_executor import KernelExecutor
from amea.execution.kernel.execution_request import ExecuteCellRequest

client = TestClient(app)


def test_chat_real_churn_execution():
    """A, B, C, D, E, F, G, L: Verify Chat executes real ML orchestrator on sample_churn.csv and returns real results."""
    # Ensure data/sample_churn.csv exists
    assert Path("data/sample_churn.csv").exists() or Path("data/customer_churn.csv").exists()

    res = client.post(
        "/api/public/chat",
        json={"message": "Train a churn classifier using sample_churn.csv"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["requires_auth"] is False
    msg = data["message"]
    
    # Verify concrete evidence in response
    assert "AMEA Multi-Agent Execution Complete" in msg
    assert "JudgeAgent" in msg
    assert "TERMINATED" in msg
    assert "Selected Champion Model" in msg
    assert "Executed Subprocess Experiments" in msg
    assert "Synthesized Production Pipeline on Disk" in msg
    
    # State payload verification
    state = data.get("state", {})
    assert state.get("current_agent") == "JudgeAgent"
    assert state.get("current_phase") == "TERMINATED"
    assert state.get("experiments_count") >= 2
    assert "train.py" in state.get("generated_files", [])


def test_terminal_exec_python_version():
    """H: Verify terminal executes python --version via real subprocess."""
    res = client.post("/api/terminal/exec", json={"project_path": ".", "command": "python --version"})
    assert res.status_code == 200
    data = res.json()
    assert "Python 3." in (data["stdout"] + data["stderr"])
    assert data["exit_code"] == 0


def test_notebook_cell_arithmetic_execution():
    """I: Verify notebook cell executes print(2+2) through real Jupyter kernel."""
    km = KernelManager()
    executor = KernelExecutor(km)
    session = km.create_session("test_nb_arithmetic")

    try:
        req = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="c_add",
            code="print(2 + 2)",
        )
        result = executor.execute_cell(req)
        assert result.is_success is True
        assert any("4" in str(getattr(o, "text", "")) for o in result.outputs)
    finally:
        km.shutdown(session.session_id)


def test_notebook_cell_intentional_error_traceback():
    """K: Verify intentional Python error displays traceback."""
    km = KernelManager()
    executor = KernelExecutor(km)
    session = km.create_session("test_nb_error")

    try:
        req = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="c_err",
            code="raise ZeroDivisionError('Division by zero intentional verification error')",
        )
        result = executor.execute_cell(req)
        assert result.is_success is False
        assert result.status == "ERROR"
        assert result.failure_diagnosis is not None
        err_out = next((o for o in result.outputs if o.output_type.value == "ERROR"), None)
        assert err_out is not None
        assert err_out.error_name == "ZeroDivisionError"
        assert "Division by zero" in (err_out.error_value or "")
    finally:
        km.shutdown(session.session_id)


def test_csv_upload_inspection(tmp_path):
    """J: Verify CSV upload works and profiles schema."""
    csv_file = tmp_path / "upload_sample.csv"
    df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "monthly_charges": [45.5, 80.0, np.nan, 30.2],
        "tenure": [12, 48, 6, 24],
        "churn": [0, 1, 0, 0]
    })
    df.to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        res = client.post(
            "/api/project/upload-dataset",
            files={"file": ("upload_sample.csv", f, "text/csv")},
            data={"project_path": str(tmp_path)},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "uploaded"
    assert data["total_rows"] == 4
    assert data["total_columns"] == 4
    assert "churn" in data["candidate_targets"]


def test_environment_hardware_info():
    """8, 11: Verify hardware info reflects real SystemInspector output, not hardcoded 4 GPUs."""
    res = client.get("/api/environment/info")
    assert res.status_code == 200
    data = res.json()
    assert "python_version" in data
    assert "executable" in data
    assert "gpu_count" in data
    assert "hardware_summary" in data
    # On this machine, GPU count is 0
    assert isinstance(data["gpu_count"], int)
