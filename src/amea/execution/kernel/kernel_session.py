"""Kernel session models, states, and runtime resource metrics."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class KernelStatus(str, Enum):
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    INTERRUPTING = "INTERRUPTING"
    RESTARTING = "RESTARTING"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"


class KernelSession(BaseModel):
    """Tracks state and metadata for an isolated interactive Python execution session."""
    session_id: str
    project_id: str
    kernel_id: Optional[str] = None
    status: KernelStatus = KernelStatus.STARTING
    workspace_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_count: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    active_cell_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
