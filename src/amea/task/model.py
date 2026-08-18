"""Explicit task models, status, dependencies, and policies."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskPriority(int, Enum):
    HIGH = 10
    NORMAL = 20
    LOW = 30


class ResourceRequirement(BaseModel):
    """Execution resources required for a specific task."""
    cpu_cores: int = 1
    ram_mb: int = 1024
    gpu_count: int = 0
    timeout_seconds: int = 300


class RetryPolicy(BaseModel):
    """Configurable retry logic for tasks."""
    max_retries: int = 2
    backoff_factor_seconds: float = 1.0
    retry_on_oom: bool = True
    retry_count: int = 0


class Task(BaseModel):
    """A discrete, executable unit of work with explicit dependencies and contracts."""
    task_id: str
    name: str
    required_capability: str
    assigned_provider: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: Set[str] = Field(default_factory=set)
    resources: ResourceRequirement = Field(default_factory=ResourceRequirement)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runtime_seconds: float = 0.0

    def is_ready(self, completed_task_ids: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.status == TaskStatus.PENDING and self.dependencies.issubset(completed_task_ids)
