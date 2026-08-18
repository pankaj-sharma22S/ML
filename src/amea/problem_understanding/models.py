"""Models and schemas for Problem Understanding Agent."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.core.state import MLTaskSpecification, TaskType


class IntentCategory(str, Enum):
    PREDICTION = "PREDICTION"
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    FORECASTING = "FORECASTING"
    CLUSTERING = "CLUSTERING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    GENERAL_ANALYSIS = "GENERAL_ANALYSIS"


class IntentAnalysis(BaseModel):
    """Extracted intent from natural language user prompt."""
    raw_query: str
    primary_intent: IntentCategory
    mentioned_target_candidate: Optional[str] = None
    requested_metrics: List[str] = Field(default_factory=list)
    stated_constraints: List[str] = Field(default_factory=list)
    domain_keywords: List[str] = Field(default_factory=list)


class ConflictFinding(BaseModel):
    """Discrepancy detected between user intent and empirical data evidence."""
    conflict_type: str  # "task_type_mismatch", "missing_target", "temporal_mismatch"
    severity: str      # "BLOCKING", "WARNING", "INFORMATIONAL"
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    resolution: str


class ProblemUnderstandingReport(BaseModel):
    """Authoritative ML problem formulation emitted for Central Orchestrator."""
    task_spec: MLTaskSpecification
    intent_analysis: IntentAnalysis
    conflicts: List[ConflictFinding] = Field(default_factory=list)
    identified_gaps: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)
    validation_strategy: str = "StratifiedKFold"
    is_feasible: bool = True
    blocking_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
