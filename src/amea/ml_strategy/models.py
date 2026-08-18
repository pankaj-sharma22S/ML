"""Models and Pydantic v2 contracts for the ML Strategy Agent."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.core.state import (
    DataProfile,
    ExperimentConfiguration,
    MLTaskSpecification,
    RegisteredExperimentRecord,
    TaskType,
)


class StrategyStatus(str, Enum):
    READY = "READY"                      # Fully formulated strategy ready for dispatch
    CONDITIONAL = "CONDITIONAL"          # Strategy formulated with stated assumptions/caveats
    BLOCKED = "BLOCKED"                  # Cannot proceed due to missing critical data or failed gates


class ModelFamily(str, Enum):
    BASELINE_DUMMY = "BaselineDummy"
    LINEAR_MODEL = "LinearModel"
    TREE_MODEL = "TreeModel"
    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    TABULAR_NEURAL_NET = "TabularNeuralNet"


class ModelCandidate(BaseModel):
    """A proposed model candidate justified by evidence."""
    candidate_id: str
    model_family: ModelFamily
    model_class_name: str
    rationale: str
    default_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 1  # 1 = Highest


class ValidationStrategySpec(BaseModel):
    """Detailed cross-validation scheme specification."""
    cv_scheme: str  # "StratifiedKFold", "KFold", "TimeSeriesSplit", "GroupKFold"
    n_splits: int = 5
    shuffle: bool = True
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    rationale: str = ""


class MetricSpecification(BaseModel):
    """Primary and secondary evaluation metrics and optimization direction."""
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    optimization_direction: str = "maximize"
    hard_constraints: Dict[str, float] = Field(default_factory=dict)


class FeatureEngineeringHypothesis(BaseModel):
    """A specific feature engineering transformation hypothesis."""
    hypothesis_id: str
    transformation_name: str
    target_features: List[str] = Field(default_factory=list)
    expected_benefit: str
    risk_factor: str
    validation_method: str
    priority: int = 1


class ExperimentSpecification(BaseModel):
    """A discrete, runnable experiment specification."""
    experiment_id: str
    hypothesis: str
    model_family: ModelFamily
    model_class_name: str
    preprocessing_steps: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    parallel_group_id: str = "group_1"
    dependencies: List[str] = Field(default_factory=list)
    timeout_seconds: int = 120
    priority: int = 1


class ExcludedApproach(BaseModel):
    """An approach deliberately excluded with mathematical justification."""
    family_or_technique: str
    reason_for_exclusion: str


class ExperimentBudgetSpec(BaseModel):
    """Resource and trial budget allocated for experimentation."""
    max_experiments: int = 5
    max_parallel_workers: int = 4
    timeout_per_experiment_sec: int = 120
    cpu_limit_per_worker: int = 2
    memory_limit_mb_per_worker: int = 2048


class StrategyConfidence(BaseModel):
    """Evidence-based confidence score and supporting factors."""
    score: float = 1.0  # 0.0 to 1.0
    factors: List[str] = Field(default_factory=list)


class MLStrategyContext(BaseModel):
    """Structured context passed to MLStrategyAgent from Central Orchestrator."""
    task_spec: MLTaskSpecification
    data_profile: Optional[DataProfile] = None
    data_quality_report: Optional[Dict[str, Any]] = None
    eda_findings: List[str] = Field(default_factory=list)
    experiment_history: List[RegisteredExperimentRecord] = Field(default_factory=list)
    budget: Optional[ExperimentBudgetSpec] = None


class MLStrategyPlan(BaseModel):
    """Authoritative ML Strategy Plan emitted for the Central Orchestrator."""
    strategy_id: str
    strategy_status: StrategyStatus
    problem_summary: str
    task_type: TaskType
    metric_spec: MetricSpecification
    validation_strategy: ValidationStrategySpec
    model_candidates: List[ModelCandidate] = Field(default_factory=list)
    baseline_candidate: Optional[ModelCandidate] = None
    feature_hypotheses: List[FeatureEngineeringHypothesis] = Field(default_factory=list)
    experiment_plan: List[ExperimentSpecification] = Field(default_factory=list)
    excluded_approaches: List[ExcludedApproach] = Field(default_factory=list)
    budget: ExperimentBudgetSpec
    stopping_criteria: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    confidence: StrategyConfidence
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
