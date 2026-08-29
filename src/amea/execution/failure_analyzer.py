"""ML Execution Failure Diagnosis & Error Classification Engine."""

import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TIMEOUT = "TIMEOUT"
    ML_METRICS_INVALID = "ML_METRICS_INVALID"
    MISSING_ARTIFACTS = "MISSING_ARTIFACTS"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    VALUE_ERROR = "VALUE_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    KEY_ERROR = "KEY_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    SUCCESS = "SUCCESS"


class FailureDiagnosis(BaseModel):
    """Structured diagnosis of an execution outcome."""
    category: FailureCategory
    root_cause: str
    traceback_summary: Optional[str] = None
    recovery_hint: str
    is_retryable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class ExecutionFailureAnalyzer:
    """Classifies execution exceptions and validates ML training artifacts and metrics."""

    @classmethod
    def diagnose(
        cls,
        exit_code: int,
        stdout: str,
        stderr: str,
        metrics: Dict[str, float],
        artifacts_created: List[str],
        primary_metric_name: Optional[str] = None,
        expected_artifacts: Optional[List[str]] = None,
        workspace_dir: Optional[Path] = None,
    ) -> FailureDiagnosis:
        """Analyze full execution evidence to classify outcome and recovery action."""

        # 1. Check for Security Violations
        if "Security violation:" in stderr or "Code security violation:" in stderr or "Dependency security violation:" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.SECURITY_VIOLATION,
                root_cause="Execution blocked by security policy.",
                traceback_summary=stderr[:300],
                recovery_hint="Modify model code/dependencies to avoid restricted modules, calls, or paths.",
                is_retryable=False,
            )

        # 2. Check for Timeout
        if exit_code == -1 or "TimeoutError" in stderr or "[EXECUTION TIMEOUT EXPIRED]" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.TIMEOUT,
                root_cause="Process exceeded configured execution timeout limit.",
                traceback_summary=stderr[:300],
                recovery_hint="Reduce dataset sample size, decrease model iterations, or allocate higher compute timeout.",
                is_retryable=True,
            )

        # 3. Check for Non-zero Exit Code & Exceptions
        if exit_code != 0:
            return cls._classify_runtime_error(stderr)

        # 4. Check Machine-Readable Metrics Validity
        metric_diag = cls._validate_metrics(metrics, primary_metric_name)
        if metric_diag:
            return metric_diag

        # 5. Check Required Model Artifacts (if expected)
        if expected_artifacts:
            for exp_art in expected_artifacts:
                if not any(a.endswith(exp_art) for a in artifacts_created):
                    if workspace_dir and not (workspace_dir / exp_art).exists():
                        return FailureDiagnosis(
                            category=FailureCategory.MISSING_ARTIFACTS,
                            root_cause=f"Script completed with exit code 0 but produced no expected artifact '{exp_art}'.",
                            recovery_hint="Ensure training script saves the required artifact to disk.",
                            is_retryable=True,
                        )

        return FailureDiagnosis(
            category=FailureCategory.SUCCESS,
            root_cause="Execution succeeded with valid metrics and artifacts.",
            recovery_hint="Proceed to model evaluation.",
            is_retryable=False,
        )

    @classmethod
    def _classify_runtime_error(cls, stderr: str) -> FailureDiagnosis:
        """Classify Python runtime traceback."""
        if "SyntaxError" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.SYNTAX_ERROR,
                root_cause="Python script contains invalid syntax.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Fix syntax errors in code synthesis.",
                is_retryable=False,
            )

        if "ModuleNotFoundError" in stderr or "No module named" in stderr:
            match = re.search(r"No module named ['\"](.*?)['\"]", stderr)
            mod_name = match.group(1) if match else "unknown"
            return FailureDiagnosis(
                category=FailureCategory.MODULE_NOT_FOUND,
                root_cause=f"Required package/module '{mod_name}' is not installed in the environment.",
                traceback_summary=stderr[:300],
                recovery_hint=f"Verify if '{mod_name}' is in the approved package allowlist and install it.",
                is_retryable=False,
            )

        if "ImportError" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.IMPORT_ERROR,
                root_cause="Import error encountered during execution.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Check symbol names and library versions.",
                is_retryable=False,
            )

        if "MemoryError" in stderr or "out of memory" in stderr.lower():
            return FailureDiagnosis(
                category=FailureCategory.RESOURCE_LIMIT,
                root_cause="Process exhausted available RAM.",
                traceback_summary=stderr[:300],
                recovery_hint="Enable data chunking, reduce batch size, or downcast numeric columns.",
                is_retryable=True,
            )

        if "inconsistent numbers of samples" in stderr or "shapes" in stderr.lower():
            return FailureDiagnosis(
                category=FailureCategory.SHAPE_MISMATCH,
                root_cause="Feature matrix X and target y dimension mismatch.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Ensure split features and target maintain matching indices and row counts.",
                is_retryable=True,
            )

        if "ValueError" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.VALUE_ERROR,
                root_cause="ValueError encountered during pipeline execution.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Inspect data types, null values, or hyperparameter ranges.",
                is_retryable=True,
            )

        if "TypeError" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.TYPE_ERROR,
                root_cause="TypeError encountered during execution.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Check feature datatypes and function argument signatures.",
                is_retryable=True,
            )

        if "FileNotFoundError" in stderr:
            return FailureDiagnosis(
                category=FailureCategory.FILE_NOT_FOUND,
                root_cause="Referenced dataset or artifact file does not exist.",
                traceback_summary=cls._extract_last_traceback_line(stderr),
                recovery_hint="Verify dataset file path in workspace.",
                is_retryable=False,
            )

        return FailureDiagnosis(
            category=FailureCategory.RUNTIME_EXCEPTION,
            root_cause="Subprocess exited with non-zero exit code.",
            traceback_summary=cls._extract_last_traceback_line(stderr),
            recovery_hint="Inspect traceback and adjust model execution script.",
            is_retryable=True,
        )

    @classmethod
    def _validate_metrics(
        cls,
        metrics: Dict[str, float],
        primary_metric_name: Optional[str] = None,
    ) -> Optional[FailureDiagnosis]:
        """Verify that emitted metrics are non-empty, finite, and within valid mathematical bounds."""
        if not metrics:
            return FailureDiagnosis(
                category=FailureCategory.ML_METRICS_INVALID,
                root_cause="Script exited with code 0 but emitted no valid __AMEA_METRICS__ payload.",
                recovery_hint="Ensure script computes validation metrics and prints JSON payload to stdout.",
                is_retryable=True,
            )

        # Check for NaN / Inf
        for k, v in metrics.items():
            if math.isnan(v) or math.isinf(v):
                return FailureDiagnosis(
                    category=FailureCategory.ML_METRICS_INVALID,
                    root_cause=f"Metric '{k}' evaluated to non-finite value ({v}).",
                    recovery_hint="Check for zero-division, unhandled nulls, or constant predictions.",
                    is_retryable=True,
                )

            # Bounded metrics check (e.g. accuracy, roc_auc, f1 should be between 0.0 and 1.0)
            if k in ["accuracy", "roc_auc", "f1", "precision", "recall"]:
                if v < 0.0 or v > 1.0:
                    return FailureDiagnosis(
                        category=FailureCategory.ML_METRICS_INVALID,
                        root_cause=f"Probability/Ratio metric '{k}' has invalid value {v} outside [0.0, 1.0].",
                        recovery_hint="Ensure metric calculation uses standard scikit-learn scoring formulas.",
                        is_retryable=True,
                    )

        if primary_metric_name and primary_metric_name not in metrics:
            return FailureDiagnosis(
                category=FailureCategory.ML_METRICS_INVALID,
                root_cause=f"Specified primary metric '{primary_metric_name}' missing from emitted metrics {list(metrics.keys())}.",
                recovery_hint=f"Ensure script computes '{primary_metric_name}'.",
                is_retryable=True,
            )

        return None

    @staticmethod
    def _extract_last_traceback_line(stderr: str) -> str:
        lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return stderr[:200]
