"""Models and Pydantic v2 schemas for the Code Synthesis Agent."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.core.state import (
    DataProfile,
    JudgeDecision,
    MLTaskSpecification,
    RegisteredExperimentRecord,
)
from amea.data_cleaning.models import CleanedDataArtifact
from amea.eda.models import EDAReport
from amea.ml_strategy.models import MLStrategyPlan


class CodeValidationReport(BaseModel):
    """AST syntax and semantic alignment validation report."""
    is_valid_syntax: bool = True
    syntax_errors: Dict[str, str] = Field(default_factory=dict)
    model_class_matched: bool = True
    hyperparameters_matched: bool = True
    target_column_matched: bool = True
    metric_matched: bool = True
    validation_notes: List[str] = Field(default_factory=list)


class CodeSynthesisContext(BaseModel):
    """Input contract for the Code Synthesis Agent received from the Orchestrator."""
    task_spec: MLTaskSpecification
    data_profile: Optional[DataProfile] = None
    eda_report: Optional[EDAReport] = None
    cleaned_data_artifact: Optional[CleanedDataArtifact] = None
    strategy_plan: Optional[MLStrategyPlan] = None
    best_candidate: RegisteredExperimentRecord
    judge_decision: Optional[JudgeDecision] = None
    feature_lineage: List[str] = Field(default_factory=list)
    execution_constraints: Dict[str, Any] = Field(default_factory=dict)


class GeneratedCodeArtifacts(BaseModel):
    """Output contract containing the complete, verified, standalone production code pipeline."""
    pipeline_id: str
    files: Dict[str, str] = Field(default_factory=dict)  # filename -> code content
    pipeline_dir: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_report: CodeValidationReport = Field(default_factory=CodeValidationReport)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
