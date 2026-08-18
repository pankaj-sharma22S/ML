"""Experiment execution models, schemas, and result containers."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"


class ResourceUsage(BaseModel):
    """Resource consumption during execution."""
    cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    duration_seconds: float = 0.0


class ExecutionError(BaseModel):
    """Structured error details when execution fails."""
    error_type: str  # "timeout", "syntax_error", "memory_limit", "runtime_error"
    message: str
    traceback: Optional[str] = None


class ModelExecutionConfiguration(BaseModel):
    """Executable experiment configuration produced by a Model Specialist."""
    experiment_id: str
    model_family: str
    model_class_name: str
    script_content: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: List[str] = Field(default_factory=list)
    dataset_path: str
    target_column: str
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    task_type: str
    seed: int = 42
    timeout_seconds: int = 120
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentResult(BaseModel):
    """Authoritative result returned by the ExperimentRunner after actual execution."""
    experiment_id: str
    status: ExperimentStatus
    model_family: str
    model_class_name: str
    cv_metrics_mean: Dict[str, float] = Field(default_factory=dict)
    cv_metrics_std: Dict[str, float] = Field(default_factory=dict)
    train_metrics_mean: Dict[str, float] = Field(default_factory=dict)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    inference_latency_ms: float = 1.0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: Optional[ExecutionError] = None
    workspace_dir: str = ""
    artifact_paths: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
