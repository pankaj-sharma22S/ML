"""Pydantic v2 request and response schemas for the Query-First Data Analysis pipeline."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """Interpreted analytical intents, target metrics, and dimensions from natural-language query."""
    primary_intent: str = "aggregation"  # trend_analysis, ranking, correlation, distribution, comparison, aggregation, anomaly_detection
    secondary_intents: List[str] = Field(default_factory=list)
    target_metrics: List[str] = Field(default_factory=list)
    target_dimensions: List[str] = Field(default_factory=list)
    is_ambiguous: bool = False
    clarification_question: Optional[str] = None
    confidence: float = 1.0


class DatasetProfile(BaseModel):
    """Statistical and structural metadata of an ingested dataset."""
    dataset_id: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    candidate_keys: List[str] = Field(default_factory=list)
    duplicate_rows_count: int = 0
    missing_summary: Dict[str, int] = Field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class DataQualityIssue(BaseModel):
    """Detected data quality issue with its impact on the user query."""
    dataset_id: str
    issue_type: str  # missing_values, duplicate_rows, invalid_type, extreme_outliers
    affected_columns: List[str] = Field(default_factory=list)
    affected_rows_count: int = 0
    severity: str = "LOW"  # CRITICAL, IMPORTANT, LOW
    impacts_user_query: bool = False
    description: str


class CleaningAction(BaseModel):
    """Audit log of an evidence-based cleaning operation."""
    dataset_id: str
    issue: str
    affected_columns: List[str]
    affected_rows: int
    operation: str
    reason: str
    before_stats: Dict[str, Any] = Field(default_factory=dict)
    after_stats: Dict[str, Any] = Field(default_factory=dict)


class InsightItem(BaseModel):
    """Calculated, evidence-backed factual insight."""
    insight: str
    evidence: str
    metric: Optional[str] = None
    dimension: Optional[str] = None
    calculation: Dict[str, Any] = Field(default_factory=dict)
    affected_dataset: Optional[str] = None
    confidence: float = 1.0


class PatternItem(BaseModel):
    """Detected statistical pattern (trend, correlation, concentration, anomaly)."""
    pattern_type: str  # trend, correlation, dominance, outlier
    description: str
    strength: float = 1.0
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RelationshipItem(BaseModel):
    """Relationship between datasets or within numeric/categorical dimensions."""
    source_a: str
    column_a: str
    source_b: str
    column_b: str
    relationship_type: str  # shared_identifier, correlation, schema_match
    strength: float = 1.0
    evidence: str


class VisualizationArtifact(BaseModel):
    """Reference to a generated query-relevant visualization."""
    id: str
    chart_type: str  # line_chart, bar_chart, correlation_heatmap, scatter_plot, box_plot, relationship_graph
    title: str
    reason: str
    artifact_path: str
    columns_visualized: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryAnalysisRequest(BaseModel):
    """API request payload for query-first data analysis."""
    query: str
    file_paths: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class QueryAnalysisResponse(BaseModel):
    """API structured analytical response."""
    run_id: str
    query: str
    query_intent: QueryIntent
    datasets: List[DatasetProfile] = Field(default_factory=list)
    data_quality: List[DataQualityIssue] = Field(default_factory=list)
    cleaning_actions: List[CleaningAction] = Field(default_factory=list)
    relationships: List[RelationshipItem] = Field(default_factory=list)
    insights: List[InsightItem] = Field(default_factory=list)
    patterns: List[PatternItem] = Field(default_factory=list)
    visualizations: List[VisualizationArtifact] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    clarification_required: bool = False
    clarification_question: Optional[str] = None
