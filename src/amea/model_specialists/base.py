"""Base protocol and capability models for Model Specialist Agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.core.state import MLTaskSpecification, TaskType
from amea.experiments.models import ModelExecutionConfiguration
from amea.ml_strategy.models import ExperimentSpecification, ModelFamily


class ModelCapability(BaseModel):
    """Declared capabilities, strengths, and resource profiles of a Model Specialist."""
    model_family: ModelFamily
    supported_task_types: List[TaskType]
    supported_data_types: List[str] = Field(default_factory=lambda: ["numeric", "categorical", "tabular"])
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    requires_scaling: bool = False
    handles_missing_values_natively: bool = False
    gpu_accelerated: bool = False


class SpecialistValidationResult(BaseModel):
    """Compatibility validation result emitted by a Model Specialist."""
    is_compatible: bool = True
    status: str = "COMPATIBLE"  # "COMPATIBLE" or "INCOMPATIBLE"
    rejection_reason: Optional[str] = None
    required_action: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ModelSpecialistBase(ABC):
    """Abstract base class for all capability-isolated Model Specialist Agents."""

    @abstractmethod
    def capabilities(self) -> ModelCapability:
        """Return the declared capability profile of this specialist."""
        pass

    @abstractmethod
    def validate_experiment(
        self,
        exp_spec: ExperimentSpecification,
        task_spec: MLTaskSpecification,
    ) -> SpecialistValidationResult:
        """Validate whether this specialist supports the requested experiment configuration."""
        pass

    @abstractmethod
    def prepare_execution(
        self,
        exp_spec: ExperimentSpecification,
        task_spec: MLTaskSpecification,
        dataset_path: str,
    ) -> ModelExecutionConfiguration:
        """Compile a standalone, runnable experiment execution configuration."""
        pass
