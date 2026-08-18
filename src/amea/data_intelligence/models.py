"""Data Intelligence models, evidence packages, and diagnostic findings."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MissingnessMechanism(str, Enum):
    MCAR = "MCAR"  # Missing Completely at Random
    MAR = "MAR"    # Missing at Random (correlated with other features)
    MNAR = "MNAR"  # Missing Not at Random (high concentration / extreme values)
    UNKNOWN = "UNKNOWN"


class MissingnessFinding(BaseModel):
    """Analysis of missingness for a single feature."""
    column_name: str
    missing_count: int
    missing_ratio: float
    candidate_mechanism: MissingnessMechanism = MissingnessMechanism.UNKNOWN
    correlated_missing_columns: List[str] = Field(default_factory=list)
    recommendation: str = ""


class OutlierFinding(BaseModel):
    """Candidate outlier diagnosis for a numeric column."""
    column_name: str
    iqr_outlier_count: int
    zscore_outlier_count: int
    outlier_ratio: float
    skewness: float
    kurtosis: float
    is_severe: bool = False
    evidence_note: str = ""


class LeakageRiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class LeakageFinding(BaseModel):
    """Leakage diagnosis for a feature or column."""
    column_name: str
    risk_level: LeakageRiskLevel
    reason: str
    target_correlation: Optional[float] = None
    mutual_info_score: Optional[float] = None
    is_identifier_candidate: bool = False
    recommended_action: str = ""


class RelationshipFinding(BaseModel):
    """Discovered relationship between features."""
    feature_a: str
    feature_b: str
    relationship_type: str  # "linear_collinear", "categorical_associated", "non_linear"
    strength: float  # Pearson R, Cramer's V, or mutual info
    description: str


class DataTreatmentCandidate(BaseModel):
    """Evidence-backed candidate treatment strategy."""
    strategy_id: str
    target_columns: List[str]
    treatment_type: str  # "imputation", "scaling", "encoding", "outlier_handling", "drop_feature"
    proposed_transformer: str
    rationale: str
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    expected_impact: str = ""


class QualityAuditReport(BaseModel):
    """Comprehensive dataset quality audit."""
    is_clean: bool
    duplicate_rows_count: int
    duplicate_rows_ratio: float
    constant_columns: List[str] = Field(default_factory=list)
    quasi_constant_columns: List[str] = Field(default_factory=list)
    missingness_findings: List[MissingnessFinding] = Field(default_factory=list)
    outlier_findings: List[OutlierFinding] = Field(default_factory=list)
    invalid_format_columns: List[str] = Field(default_factory=list)
    quality_score: float = 1.0  # 0.0 to 1.0


class DatasetVersion(BaseModel):
    """Immutable dataset version and provenance metadata."""
    dataset_id: str
    version_id: str
    source_uri: str
    source_hash_sha256: str
    schema_hash: str
    parent_version_id: Optional[str] = None
    transformation_history: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataEvidencePackage(BaseModel):
    """Structured, evidence-rich package returned to Central Orchestrator."""
    dataset_version: DatasetVersion
    total_rows: int
    total_columns: int
    memory_mb: float
    quality_audit: QualityAuditReport
    leakage_findings: List[LeakageFinding] = Field(default_factory=list)
    relationship_findings: List[RelationshipFinding] = Field(default_factory=list)
    treatment_candidates: List[DataTreatmentCandidate] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list)
    summary_findings: List[str] = Field(default_factory=list)
