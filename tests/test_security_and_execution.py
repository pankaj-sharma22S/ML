"""Unit tests for Sandbox Security and Subprocess Execution."""

from pathlib import Path
import pytest
from amea.core.exceptions import SecurityViolationError
from amea.execution.security import SecurityBoundary
from amea.execution.subprocess_executor import SubprocessExecutor


def test_security_boundary_path_validation(tmp_path):
    boundary = SecurityBoundary(allowed_root=tmp_path)

    # Valid child path
    valid_path = tmp_path / "subdir" / "file.py"
    assert boundary.validate_path(valid_path) == valid_path.resolve()

    # Path traversal attack
    escaping_path = tmp_path / ".." / "outside.py"
    with pytest.raises(SecurityViolationError):
        boundary.validate_path(escaping_path)


def test_security_boundary_command_blocking(tmp_path):
    boundary = SecurityBoundary(allowed_root=tmp_path)

    # Blocked shell injection
    with pytest.raises(SecurityViolationError):
        boundary.validate_command("rm -rf /")

    with pytest.raises(SecurityViolationError):
        boundary.validate_command("format c:")


def test_security_boundary_env_sanitization(tmp_path):
    boundary = SecurityBoundary(allowed_root=tmp_path)
    dirty_env = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-secret-key-12345",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "NORMAL_VAR": "hello",
    }

    clean_env = boundary.sanitize_environment(base_env=dirty_env)
    assert "OPENAI_API_KEY" not in clean_env
    assert "AWS_SECRET_ACCESS_KEY" not in clean_env
    assert clean_env["NORMAL_VAR"] == "hello"


def test_subprocess_executor_execution(tmp_path):
    executor = SubprocessExecutor(sandbox_root=tmp_path)
    script = """
import json
print("Worker running...")
metrics = {"accuracy": 0.95}
print(f"__AMEA_METRICS__={json.dumps(metrics)}")
"""
    res = executor.execute_script(run_id="test_run_01", script_content=script, timeout_seconds=10)
    assert res.is_success
    assert res.exit_code == 0
    assert "accuracy" in res.metrics_extracted
    assert res.metrics_extracted["accuracy"] == 0.95
