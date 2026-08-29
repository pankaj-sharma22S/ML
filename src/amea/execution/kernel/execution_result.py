"""Execution results, structured outputs, DataFrames, images, and diagnostics."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.execution.failure_analyzer import FailureDiagnosis


class CellOutputType(str, Enum):
    TEXT = "TEXT"
    STREAM = "STREAM"
    DATAFRAME = "DATAFRAME"
    SCALAR = "SCALAR"
    IMAGE = "IMAGE"
    HTML = "HTML"
    JSON = "JSON"
    ERROR = "ERROR"


class DataFramePreview(BaseModel):
    """Structured representation of a DataFrame for interactive frontend tables."""
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    rows_preview_count: int = 0
    total_rows: int = 0
    total_columns: int = 0
    data: List[Dict[str, Any]] = Field(default_factory=list)
    is_truncated: bool = False


class CellOutput(BaseModel):
    """A single discrete output produced by cell execution."""
    output_type: CellOutputType
    text: Optional[str] = None
    stream_name: Optional[str] = None  # "stdout" or "stderr"
    dataframe: Optional[DataFramePreview] = None
    image_artifact_path: Optional[str] = None
    image_base64: Optional[str] = None
    scalar_value: Optional[Any] = None
    data: Optional[Dict[str, Any]] = None
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    traceback: Optional[List[str]] = None


class CellExecutionResult(BaseModel):
    """Complete execution result of a single notebook/editor cell."""
    session_id: str
    cell_id: str
    status: str = "SUCCESS"  # SUCCESS, ERROR, TIMEOUT, SECURITY_BLOCKED, INTERRUPTED
    execution_count: Optional[int] = None
    outputs: List[CellOutput] = Field(default_factory=list)
    duration_ms: float = 0.0
    failure_diagnosis: Optional[FailureDiagnosis] = None
    is_success: bool = True
