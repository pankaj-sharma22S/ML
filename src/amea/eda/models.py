"""Structured models and finding schemas for the EDA & Data Insight Agent."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EDASeverity(str, Enum):
    CRITICAL = "CRITICAL"          # Blocks or severely compromises modeling without intervention
    IMPORTANT = "IMPORTANT"        # Materially affects model performance or validation validity
    MINOR = "MINOR"                # Minor nuisance; standard model defaults can handle safely
    INFORMATIONAL = "INFORMATIONAL"  # Descriptive insight for feature engineering/interpretability


class OutlierCategory(str, Enum):
    LIKELY_INVALID = "LIKELY_INVALID"          # Data entry error / physically impossible value
    POTENTIALLY_INVALID = "POTENTIALLY_INVALID"  # Extreme statistical anomaly; needs validation
    LEGITIMATE_EXTREME = "LEGITIMATE_EXTREME"    # Valid heavy-tailed natural distribution
    UNCERTAIN = "UNCERTAIN"                    # Insufficient data to determine validity


class EDAFinding(BaseModel):
    """Actionable structured finding emitted by the EDA Agent."""
    finding_id: str
    category: str  # "distribution", "outlier", "categorical", "target", "temporal", "relationship", "leakage"
    feature_name: Optional[str] = None
    observation: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    ml_impact: str
    severity: EDASeverity
    suggested_investigation: str = ""
    candidate_strategies: List[str] = Field(default_factory=list)
    requires_validation: bool = False


class DistributionFinding(BaseModel):
    """Numeric distribution shape and tail analysis."""
    column_name: str
    mean: float
    median: float
    std: float
    iqr: float
    skewness: float
    kurtosis: float
    is_zero_inflated: bool = False
    zero_ratio: float = 0.0
    is_heavy_tailed: bool = False
    distribution_shape: str = "normal"  # "symmetric", "right_skewed", "left_skewed", "bimodal", "constant"


class CategoricalFinding(BaseModel):
    """Categorical column frequency and cardinality analysis."""
    column_name: str
    distinct_count: int
    cardinality_ratio: float
    dominant_category: Optional[str] = None
    dominant_ratio: float = 0.0
    rare_categories_count: int = 0
    rare_categories_ratio: float = 0.0
    has_high_cardinality: bool = False


class TargetAnalysisFinding(BaseModel):
    """Target variable distribution and balance analysis."""
    target_column: str
    task_type: str
    is_imbalanced: bool = False
    imbalance_ratio: Optional[float] = None
    class_distribution: Optional[Dict[str, float]] = None
    minority_class_count: Optional[int] = None
    skewness: Optional[float] = None
    is_zero_inflated: bool = False
    target_summary: str = ""


class TemporalFinding(BaseModel):
    """Temporal structure, ordering, and split hazard analysis."""
    datetime_column: str
    is_monotonic_increasing: bool = True
    has_temporal_gaps: bool = False
    temporal_span_days: Optional[float] = None
    split_hazard_warning: bool = False
    recommendation: str = ""


class EDAReport(BaseModel):
    """Comprehensive EDA Evidence Package returned to Central Orchestrator."""
    dataset_name: str
    total_rows: int
    total_columns: int
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    findings: List[EDAFinding] = Field(default_factory=list)
    distributions: List[DistributionFinding] = Field(default_factory=list)
    categoricals: List[CategoricalFinding] = Field(default_factory=list)
    target_analysis: Optional[TargetAnalysisFinding] = None
    temporal_analysis: Optional[TemporalFinding] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
