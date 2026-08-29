"""Subprocess-based isolated executor with AST security, dependency validation, and ML failure diagnosis."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from amea.core.exceptions import SecurityViolationError
from amea.execution.executor import Executor, ExecutionResult
from amea.execution.failure_analyzer import ExecutionFailureAnalyzer, FailureCategory, FailureDiagnosis
from amea.execution.security import SecurityBoundary
from amea.execution.workspace import IsolatedWorkspace


class SubprocessExecutor(Executor):
    """Executes Python code in an isolated subprocess within a sandboxed directory."""

    def __init__(self, sandbox_root: Path, security_boundary: Optional[SecurityBoundary] = None):
        self.sandbox_root = sandbox_root.resolve()
        self.security = security_boundary or SecurityBoundary(allowed_root=self.sandbox_root)
        self.failure_analyzer = ExecutionFailureAnalyzer()

    def execute_script(
        self,
        run_id: str,
        script_content: str,
        timeout_seconds: int = 300,
        additional_files: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        primary_metric_name: Optional[str] = None,
        expected_artifacts: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script inside an ephemeral isolated workspace."""
        start_time = time.time()
        workspace = IsolatedWorkspace(self.sandbox_root, run_id)
        workspace_dir = workspace.create()

        # 1. Security validation: Workspace path containment
        self.security.validate_path(workspace_dir)

        # 2. Security validation: Dependency allowlist / blocked packages
        if dependencies:
            try:
                self.security.validate_dependencies(dependencies)
            except SecurityViolationError as e:
                return ExecutionResult(
                    run_id=run_id,
                    exit_code=1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=round(time.time() - start_time, 3),
                    error_type="SecurityViolationError",
                    failure_diagnosis=FailureDiagnosis(
                        category=FailureCategory.SECURITY_VIOLATION,
                        root_cause=str(e),
                        recovery_hint="Use only approved ML packages.",
                    ),
                    is_success=False,
                )

        # 3. Security validation: AST Python source code inspection
        try:
            self.security.validate_code(script_content)
        except SecurityViolationError as e:
            return ExecutionResult(
                run_id=run_id,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=round(time.time() - start_time, 3),
                error_type="SecurityViolationError",
                failure_diagnosis=FailureDiagnosis(
                    category=FailureCategory.SECURITY_VIOLATION,
                    root_cause=str(e),
                    recovery_hint="Remove forbidden calls/modules (e.g. os.system, socket, subprocess, eval).",
                ),
                is_success=False,
            )

        # Write primary script
        main_script = workspace.write_file("main.py", script_content)

        # Write additional dependency files if provided
        if additional_files:
            for rel_path, content in additional_files.items():
                workspace.write_file(rel_path, content)

        # Write requirements manifest
        if dependencies:
            workspace.write_file("requirements.txt", "\n".join(dependencies))

        # 4. Sanitize environment (Two-sided secret protection)
        clean_env = self.security.sanitize_environment()

        exit_code = -1
        stdout = ""
        stderr = ""
        error_type = None
        metrics_extracted: Dict[str, float] = {}
        artifacts: List[str] = []

        try:
            cmd = [sys.executable, "-u", "main.py"]
            # Security scan of command string
            self.security.validate_command(" ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace_dir),
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
                stdout = stdout_data or ""
                stderr = stderr_data or ""
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_data, stderr_data = proc.communicate()
                stdout = stdout_data or ""
                stderr = (stderr_data or "") + "\n[EXECUTION TIMEOUT EXPIRED]"
                exit_code = -1
                error_type = "TimeoutError"

        except Exception as e:
            stderr = str(e)
            exit_code = 1
            error_type = type(e).__name__

        duration = time.time() - start_time

        # Extract metrics
        metrics_extracted = self._extract_metrics(stdout, workspace_dir)

        # Collect created files
        for item in workspace_dir.glob("**/*"):
            if item.is_file() and item.name != "main.py":
                artifacts.append(str(item.relative_to(workspace_dir)))

        # 5. ML Failure Diagnosis & Metric Validation
        diagnosis = self.failure_analyzer.diagnose(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metrics=metrics_extracted,
            artifacts_created=artifacts,
            primary_metric_name=primary_metric_name,
            expected_artifacts=expected_artifacts,
            workspace_dir=workspace_dir,
        )

        is_success = (diagnosis.category == FailureCategory.SUCCESS)

        return ExecutionResult(
            run_id=run_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 3),
            metrics_extracted=metrics_extracted,
            artifacts_created=artifacts,
            error_type=error_type or (diagnosis.category.value if not is_success else None),
            failure_diagnosis=diagnosis,
            is_success=is_success,
        )

    def _extract_metrics(self, stdout: str, workspace_dir: Path) -> Dict[str, float]:
        """Extract metrics from metrics.json file or JSON markers in stdout."""
        metrics_file = workspace_dir / "metrics.json"
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            except Exception:
                pass

        # Try to parse stdout marker e.g., __AMEA_METRICS__={"roc_auc": 0.88}
        match = re.search(r"__AMEA_METRICS__=({.*?})", stdout)
        if match:
            try:
                data = json.loads(match.group(1))
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            except Exception:
                pass

        return {}
