"""Executor interfaces and execution result models."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class ExecutionResult(BaseModel):
    """Result of an executed script or task."""
    run_id: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    artifacts_created: List[str] = Field(default_factory=list)
    metrics_extracted: Dict[str, float] = Field(default_factory=dict)
    error_type: Optional[str] = None
    is_success: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Executor(ABC):
    """Abstract interface for executing code in an isolated environment."""

    @abstractmethod
    def execute_script(
        self,
        run_id: str,
        script_content: str,
        timeout_seconds: int = 300,
        additional_files: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script inside an isolated sandbox."""
        pass
