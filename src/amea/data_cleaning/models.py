"""Models and artifact schemas for the Data Cleaning & Quality Agent."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.data_intelligence.models import DatasetVersion


class CleaningActionType(str, Enum):
    DROP_COLUMNS = "DROP_COLUMNS"
    IMPUTE_MEDIAN = "IMPUTE_MEDIAN"
    IMPUTE_MODE = "IMPUTE_MODE"
    CLIP_OUTLIERS = "CLIP_OUTLIERS"
    GROUP_RARE_CATEGORIES = "GROUP_RARE_CATEGORIES"
    STANDARDIZE_NUMERIC = "STANDARDIZE_NUMERIC"
    ROBUST_SCALE = "ROBUST_SCALE"


class CleaningAction(BaseModel):
    """A discrete cleaning operation with targeted columns and parameters."""
    action_type: CleaningActionType
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class CleaningPlan(BaseModel):
    """Ordered sequence of approved cleaning actions."""
    plan_id: str
    actions: List[CleaningAction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PostCleaningValidationReport(BaseModel):
    """Post-cleaning validation audit verifying dataset readiness."""
    is_valid: bool
    initial_rows: int
    final_rows: int
    initial_columns: int
    final_columns: int
    remaining_null_count: int
    target_column_preserved: bool
    columns_dropped: List[str] = Field(default_factory=list)
    validation_messages: List[str] = Field(default_factory=list)


class CleanedDataArtifact(BaseModel):
    """Structured artifact containing cleaned dataset pointers and provenance."""
    cleaned_dataset_path: str
    dataset_version: DatasetVersion
    validation_report: PostCleaningValidationReport
    applied_cleaning_plan: CleaningPlan
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
