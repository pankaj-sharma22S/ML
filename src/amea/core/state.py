"""Strongly typed Pydantic v2 state schemas for the Autonomous ML Engineer."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LifecyclePhase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    INSPECT = "INSPECT"
    VALIDATE = "VALIDATE"
    PLAN = "PLAN"
    DISPATCH = "DISPATCH"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    EVALUATE = "EVALUATE"
    REPLAN = "REPLAN"
    BUILD = "BUILD"
    VERIFY = "VERIFY"
    FINALIZE = "FINALIZE"
    TERMINATED = "TERMINATED"


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    UNSPECIFIED = "unspecified"


class GapSeverity(str, Enum):
    CRITICAL = "CRITICAL"    # Must stop and clarify/retrieve
    IMPORTANT = "IMPORTANT"  # Must attempt resolution or request clarification
    MINOR = "MINOR"          # Safe to continue with explicit documented assumption


class GapItem(BaseModel):
    """An identified gap in information or capability."""
    description: str
    severity: GapSeverity
    affected_components: List[str] = Field(default_factory=list)
    resolution: Optional[str] = None


class MLTaskSpecification(BaseModel):
    """Formalized machine learning task specification."""
    task_type: TaskType = TaskType.UNSPECIFIED
    target_column: Optional[str] = None
    feature_candidates: List[str] = Field(default_factory=list)
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = Field(default_factory=list)
    optimization_direction: str = "maximize"  # maximize or minimize
    max_inference_latency_ms: Optional[float] = None
    success_metric_threshold: Optional[float] = None
    random_seed: int = 42


class ColumnProfile(BaseModel):
    """Deterministic statistical profile for a single column."""
    dtype: str
    null_count: int = 0
    null_ratio: float = 0.0
    distinct_count: int = 0
    is_constant: bool = False
    is_unique: bool = False
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    class_balance: Optional[Dict[str, float]] = None


class DataProfile(BaseModel):
    """Deterministic dataset profile."""
    dataset_path: str
    dataset_sha256: str
    total_rows: int
    total_columns: int
    columns: Dict[str, ColumnProfile]
    memory_footprint_mb: float
    duplicate_rows: int = 0
    potential_leakage_columns: List[str] = Field(default_factory=list)
    is_sampled: bool = False


class ExperimentConfiguration(BaseModel):
    """Specification for an isolated experiment."""
    experiment_id: str
    model_family: str
    model_class_name: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: List[str] = Field(default_factory=list)
    cv_strategy: str = "StratifiedKFold"
    n_splits: int = 5
    early_stopping_rounds: Optional[int] = 50
    timeout_seconds: int = 300
    seed: int = 42


class RegisteredExperimentRecord(BaseModel):
    """Immutable record of an executed experiment."""
    experiment_id: str
    model_family: str
    hyperparameters: Dict[str, Any]
    cv_metrics_mean: Dict[str, float]
    cv_metrics_std: Dict[str, float]
    train_metrics_mean: Dict[str, float]
    training_duration_sec: float
    inference_latency_ms: float
    peak_memory_mb: float
    artifact_uri: Optional[str] = None
    exit_code: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationAuditReport(BaseModel):
    """Audit of experiment validity, leakage, and generalization."""
    experiment_id: str
    is_leakage_suspected: bool = False
    leakage_reasons: List[str] = Field(default_factory=list)
    overfitting_gap: float = 0.0
    is_stable: bool = True
    beats_baseline: bool = True
    audit_passed: bool = True
    confidence_score: float = 1.0


class JudgeDecision(BaseModel):
    """Objective decision emitted by Judge Agent."""
    action: str  # "ACCEPT_BEST_CANDIDATE" or "TRIGGER_IMPROVEMENT"
    selected_experiment_id: Optional[str] = None
    pareto_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str
    improvement_directives: List[str] = Field(default_factory=list)


class GeneratedCodeArtifacts(BaseModel):
    """Synthesized modular pipeline code files."""
    files: Dict[str, str] = Field(default_factory=dict)  # relative_path -> content
    entrypoint: str = "train.py"
    target_environment: str = "python"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineExecutionResult(BaseModel):
    """Validation results from running synthesized code in sandbox."""
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    execution_duration_sec: float = 0.0
    verified_metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts_created: List[str] = Field(default_factory=list)
    is_verified: bool = False


class FinalReport(BaseModel):
    """Final synthesis report delivered to the user."""
    project_id: str
    task_id: str
    summary: str
    best_model_family: str
    best_metrics: Dict[str, float]
    total_experiments_run: int
    total_duration_sec: float
    verified_code_paths: List[str] = Field(default_factory=list)
    remaining_risks: List[str] = Field(default_factory=list)


class OrchestratorDecision(BaseModel):
    """Formal decision object emitted at major routing transitions."""
    decision_id: str
    phase: LifecyclePhase
    objective: str
    known_facts: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    identified_gaps: List[GapItem] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    selected_agents: List[str] = Field(default_factory=list)
    rejected_agents: List[str] = Field(default_factory=list)
    parallel_groups: List[List[str]] = Field(default_factory=list)
    confidence_score: float = 1.0
    rationale: str
    next_action: str


class GlobalState(BaseModel):
    """Strongly typed root state for the Autonomous ML Engineer system."""
    schema_version: str = "1.0.0"
    project_id: str = "amea-project"
    task_id: str = "task-001"
    user_request: str = ""
    current_phase: LifecyclePhase = LifecyclePhase.UNDERSTAND
    iteration: int = 0
    max_iterations: int = 3

    # Task & Specifications
    task_spec: Optional[MLTaskSpecification] = None
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Data Intelligence Evidence
    data_profile: Optional[DataProfile] = None
    data_quality_report: Dict[str, Any] = Field(default_factory=dict)
    eda_findings: List[str] = Field(default_factory=list)

    # Experimentation & Tracking
    experiment_queue: List[ExperimentConfiguration] = Field(default_factory=list)
    experiment_ledger: List[RegisteredExperimentRecord] = Field(default_factory=list)
    audit_reports: Dict[str, EvaluationAuditReport] = Field(default_factory=dict)
    best_candidate: Optional[RegisteredExperimentRecord] = None
    judge_decision: Optional[JudgeDecision] = None

    # Code Synthesis & Execution
    code_artifacts: Optional[GeneratedCodeArtifacts] = None
    execution_result: Optional[PipelineExecutionResult] = None
    final_report: Optional[FinalReport] = None

    # Auditing & Reasoning Trace
    identified_gaps: List[GapItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    decisions_history: List[OrchestratorDecision] = Field(default_factory=list)
    active_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)

    # Status
    termination_reason: Optional[str] = None
    is_terminal: bool = False
