"""Comprehensive tests for Secure Subprocess, AST Security, Dependency Validation, and ML Failure Diagnosis."""

import os
from pathlib import Path
import pytest

from amea.core.exceptions import SecurityViolationError
from amea.execution.failure_analyzer import ExecutionFailureAnalyzer, FailureCategory
from amea.execution.security import (
    AstSecurityValidator,
    DependencySecurityValidator,
    EnvironmentSanitizer,
    SecurityBoundary,
)
from amea.execution.subprocess_executor import SubprocessExecutor
from amea.execution.workspace import IsolatedWorkspace
from amea.experiments.models import ModelExecutionConfiguration
from amea.experiments.runner import ExperimentRunner


# ============================================================
# 1. AST Security Validator Tests
# ============================================================

def test_ast_security_blocks_forbidden_imports():
    bad_codes = [
        "import subprocess\nsubprocess.run(['ls'])",
        "import socket\ns = socket.socket()",
        "from urllib.request import urlopen",
        "import paramiko",
        "import ctypes",
    ]
    for code in bad_codes:
        violations = AstSecurityValidator.validate_code_safety(code)
        assert len(violations) > 0, f"Expected security violation for code: {code}"


def test_ast_security_blocks_dangerous_calls_and_builtins():
    bad_codes = [
        "eval('2 + 2')",
        "exec('import os')",
        "__import__('os').system('whoami')",
        "import os\nos.system('calc.exe')",
        "import os\nos.popen('dir')",
        "import shutil\nshutil.rmtree('/')",
    ]
    for code in bad_codes:
        violations = AstSecurityValidator.validate_code_safety(code)
        assert len(violations) > 0, f"Expected security violation for code: {code}"


def test_ast_security_blocks_sensitive_paths():
    bad_codes = [
        "with open('/etc/passwd', 'r') as f: data = f.read()",
        "p = 'C:\\\\Windows\\\\System32\\\\cmd.exe'",
        "key_path = '~/.ssh/id_rsa'",
        "aws_path = '~/.aws/credentials'",
    ]
    for code in bad_codes:
        violations = AstSecurityValidator.validate_code_safety(code)
        assert len(violations) > 0, f"Expected path violation for code: {code}"


def test_ast_security_allows_safe_ml_code():
    safe_code = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

X = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([0, 1])
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, 'model.joblib')
print('__AMEA_METRICS__={"accuracy": 1.0}')
"""
    violations = AstSecurityValidator.validate_code_safety(safe_code)
    assert len(violations) == 0


# ============================================================
# 2. Dependency Security Validator Tests
# ============================================================

def test_dependency_security_validator():
    # Approved packages
    valid, violations = DependencySecurityValidator.validate_dependencies(["numpy>=1.26.0", "pandas", "scikit-learn"])
    assert valid is True
    assert len(violations) == 0

    # Blocked malicious / network packages
    valid_blocked, violations_blocked = DependencySecurityValidator.validate_dependencies(["requests", "paramiko"])
    assert valid_blocked is False
    assert any("BLOCK" in v for v in violations_blocked)

    # Unapproved packages
    valid_unapproved, violations_unapproved = DependencySecurityValidator.validate_dependencies(["django", "flask"])
    assert valid_unapproved is False
    assert any("NOT in approved" in v for v in violations_unapproved)


# ============================================================
# 3. Environment Sanitization Tests (Two-sided Protection)
# ============================================================

def test_environment_sanitizer_scrubs_secrets():
    dirty_env = {
        "OPENAI_API_KEY": "sk-12345secret",
        "AWS_SECRET_ACCESS_KEY": "secret_aws_key",
        "GITHUB_TOKEN": "ghp_tokensecret",
        "DB_PASSWORD": "supersecretpassword",
        "AUTH_HEADER": "Bearer xyz",
        "PATH": "/usr/bin:/bin",
        "USER": "sandbox_user",
    }

    clean_env = EnvironmentSanitizer.sanitize(dirty_env)
    assert "OPENAI_API_KEY" not in clean_env
    assert "AWS_SECRET_ACCESS_KEY" not in clean_env
    assert "GITHUB_TOKEN" not in clean_env
    assert "DB_PASSWORD" not in clean_env
    assert "AUTH_HEADER" not in clean_env
    assert clean_env["PATH"] == "/usr/bin:/bin"
    assert clean_env["PYTHONUNBUFFERED"] == "1"
    assert clean_env["AMEA_SANDBOX_ACTIVE"] == "1"


# ============================================================
# 4. Workspace Isolation & Path Traversal Tests
# ============================================================

def test_isolated_workspace_path_traversal_blocked(tmp_path):
    workspace = IsolatedWorkspace(base_dir=tmp_path, run_id="test_exp_1")
    workspace.create()

    # Valid write
    valid_file = workspace.write_file("sub/script.py", "print('hello')")
    assert valid_file.exists()

    # Path traversal escape attempt
    with pytest.raises(ValueError, match="Path traversal blocked"):
        workspace.write_file("../../escape.py", "print('hack')")


# ============================================================
# 5. Subprocess Execution & Failure Diagnosis Tests
# ============================================================

def test_subprocess_executor_valid_execution(tmp_path):
    executor = SubprocessExecutor(sandbox_root=tmp_path)
    script = """
import json
import joblib
from sklearn.linear_model import Ridge
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = Ridge()
model.fit(X, y)
joblib.dump(model, 'model.joblib')

metrics = {"rmse": 0.05}
print(f"__AMEA_METRICS__={json.dumps(metrics)}")
"""
    res = executor.execute_script(
        run_id="exp_valid",
        script_content=script,
        primary_metric_name="rmse",
        timeout_seconds=30,
    )

    assert res.is_success is True
    assert res.exit_code == 0
    assert "rmse" in res.metrics_extracted
    assert res.metrics_extracted["rmse"] == 0.05
    assert res.failure_diagnosis.category == FailureCategory.SUCCESS


def test_subprocess_executor_blocks_malicious_code_before_execution(tmp_path):
    executor = SubprocessExecutor(sandbox_root=tmp_path)
    malicious_script = "import os\nos.system('whoami')"

    res = executor.execute_script(
        run_id="exp_malicious",
        script_content=malicious_script,
    )

    assert res.is_success is False
    assert res.error_type == "SecurityViolationError"
    assert res.failure_diagnosis.category == FailureCategory.SECURITY_VIOLATION


def test_subprocess_executor_timeout_kill(tmp_path):
    executor = SubprocessExecutor(sandbox_root=tmp_path)
    infinite_loop = """
import time
while True:
    time.sleep(0.5)
"""
    res = executor.execute_script(
        run_id="exp_timeout",
        script_content=infinite_loop,
        timeout_seconds=1,
    )

    assert res.is_success is False
    assert res.failure_diagnosis.category == FailureCategory.TIMEOUT
    assert res.failure_diagnosis.is_retryable is True


def test_failure_analyzer_nan_metrics(tmp_path):
    diag = ExecutionFailureAnalyzer.diagnose(
        exit_code=0,
        stdout="",
        stderr="",
        metrics={"roc_auc": float("nan")},
        artifacts_created=["model.joblib"],
        primary_metric_name="roc_auc",
    )
    assert diag.category == FailureCategory.ML_METRICS_INVALID
    assert "non-finite" in diag.root_cause


def test_failure_analyzer_missing_model_artifact(tmp_path):
    workspace = tmp_path / "exp_no_artifact"
    workspace.mkdir(parents=True)

    diag = ExecutionFailureAnalyzer.diagnose(
        exit_code=0,
        stdout="",
        stderr="",
        metrics={"roc_auc": 0.85},
        artifacts_created=[],
        primary_metric_name="roc_auc",
        expected_artifacts=["model.joblib"],
        workspace_dir=workspace,
    )
    assert diag.category == FailureCategory.MISSING_ARTIFACTS


def test_experiment_runner_end_to_end(tmp_path):
    runner = ExperimentRunner(base_workspace_dir=tmp_path)
    dummy_csv = tmp_path / "data.csv"
    dummy_csv.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")

    script = f"""
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('{dummy_csv.name}')
model = LogisticRegression()
model.fit(df[['x']], df['y'])
joblib.dump(model, 'model.joblib')

metrics = {{"roc_auc": 0.90}}
print(f"__AMEA_METRICS__={{json.dumps(metrics)}}")
"""

    config = ModelExecutionConfiguration(
        experiment_id="exp_e2e_runner",
        model_family="LinearModel",
        model_class_name="LogisticRegression",
        script_content=script,
        dataset_path=str(dummy_csv),
        target_column="y",
        primary_metric="roc_auc",
        task_type="binary_classification",
        dependencies=["numpy", "pandas", "scikit-learn", "joblib"],
    )

    result = runner.run_experiment(config)
    assert result.status == "SUCCESS"
    assert result.cv_metrics_mean.get("roc_auc") == 0.90
    assert result.failure_diagnosis.category == FailureCategory.SUCCESS
