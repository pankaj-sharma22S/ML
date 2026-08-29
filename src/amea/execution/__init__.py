"""Execution sandbox, AST security, dependency validation, and resource monitoring package."""

from amea.execution.executor import Executor, ExecutionResult, WorkerStatus
from amea.execution.failure_analyzer import ExecutionFailureAnalyzer, FailureCategory, FailureDiagnosis
from amea.execution.resource import ResourceManager
from amea.execution.security import (
    AstSecurityValidator,
    DependencySecurityValidator,
    EnvironmentSanitizer,
    SecurityBoundary,
)
from amea.execution.subprocess_executor import SubprocessExecutor
from amea.execution.workspace import IsolatedWorkspace

__all__ = [
    "Executor",
    "ExecutionResult",
    "WorkerStatus",
    "ExecutionFailureAnalyzer",
    "FailureCategory",
    "FailureDiagnosis",
    "ResourceManager",
    "AstSecurityValidator",
    "DependencySecurityValidator",
    "EnvironmentSanitizer",
    "SecurityBoundary",
    "SubprocessExecutor",
    "IsolatedWorkspace",
]
