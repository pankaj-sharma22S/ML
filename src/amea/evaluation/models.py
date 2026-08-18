"""Evaluation and candidate arbitration models."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditVerdict(str, Enum):
    PASSED = "PASSED"
    OVERFITTING_WARNING = "OVERFITTING_WARNING"
    LEAKAGE_SUSPECTED = "LEAKAGE_SUSPECTED"
    UNSTABLE = "UNSTABLE"
    FAILED = "FAILED"


class CandidateAuditReport(BaseModel):
    """Detailed audit report on an individual executed candidate experiment."""
    experiment_id: str
    verdict: AuditVerdict
    primary_metric_val: float
    overfitting_gap: float
    is_leakage_suspected: bool = False
    is_stable: bool = True
    beats_baseline: bool = True
    audit_passed: bool = True
    audit_notes: List[str] = Field(default_factory=list)


class ParetoCandidateRecord(BaseModel):
    """Candidate profile evaluated for multi-objective Pareto trade-offs."""
    experiment_id: str
    model_family: str
    primary_metric: float
    latency_ms: float
    memory_mb: float
    rank: int = 1
