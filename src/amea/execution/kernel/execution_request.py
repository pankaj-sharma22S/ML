"""Interactive cell execution requests and execution modes."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    CELL = "CELL"
    FROM_HERE = "FROM_HERE"
    ALL = "ALL"


class CellType(str, Enum):
    CODE = "CODE"
    MARKDOWN = "MARKDOWN"


class NotebookCell(BaseModel):
    """Represents a notebook cell in the interactive editor."""
    cell_id: str
    cell_type: CellType = CellType.CODE
    source: str
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecuteCellRequest(BaseModel):
    """Request payload to execute a single code cell in an active kernel session."""
    session_id: str
    cell_id: str
    code: str
    execution_count: Optional[int] = None
    timeout_seconds: Optional[int] = None
    user_approved: bool = True
    mode: ExecutionMode = ExecutionMode.CELL


class BatchExecuteRequest(BaseModel):
    """Request payload to execute a series of cells sequentially."""
    session_id: str
    cells: List[NotebookCell]
    start_cell_id: Optional[str] = None
    stop_on_error: bool = True
    timeout_per_cell_seconds: Optional[int] = None
