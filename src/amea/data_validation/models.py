"""Models and schema definitions for the Data Validation & Quality Gate Agent."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QualityGateVerdict(str, Enum):
    PASSED = "PASSED"                      # All critical and important checks satisfied
    WARNING_PROCEED = "WARNING_PROCEED"    # Non-critical warnings detected; safe to proceed
    REJECTED_BLOCKING = "REJECTED_BLOCKING"  # Critical failure; cannot proceed to modeling


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class ValidationCheckResult(BaseModel):
    """Result of an individual deterministic quality check."""
    check_name: str
    category: str  # "schema", "integrity", "missingness", "distribution", "domain"
    status: CheckStatus
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    is_blocking: bool = False


class DistributionShiftMetric(BaseModel):
    """Pre vs post cleaning distribution shift metrics."""
    column_name: str
    raw_mean: float
    clean_mean: float
    mean_diff_pct: float
    raw_median: float
    clean_median: float
    raw_std: float
    clean_std: float
    ks_statistic: float
    ks_pvalue: float
    is_severely_distorted: bool = False


class QualityGateReport(BaseModel):
    """Authoritative quality gate evaluation report produced for the Central Orchestrator."""
    verdict: QualityGateVerdict
    dataset_name: str
    total_checks_run: int
    checks_passed: int
    checks_warned: int
    checks_failed: int
    check_results: List[ValidationCheckResult] = Field(default_factory=list)
    distribution_shifts: List[DistributionShiftMetric] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)
    recommendations_for_orchestrator: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
