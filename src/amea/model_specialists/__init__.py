"""Model Specialist Agents package."""

from amea.model_specialists.base import (
    ModelCapability,
    SpecialistValidationResult,
    ModelSpecialistBase,
)
from amea.model_specialists.linear_specialist import LinearSpecialistAgent
from amea.model_specialists.tree_specialist import TreeModelSpecialistAgent
from amea.model_specialists.boosting_specialist import BoostingSpecialistAgent
from amea.model_specialists.neural_specialist import NeuralSpecialistAgent
from amea.model_specialists.registry import ModelSpecialistRegistry

__all__ = [
    "ModelCapability",
    "SpecialistValidationResult",
    "ModelSpecialistBase",
    "LinearSpecialistAgent",
    "TreeModelSpecialistAgent",
    "BoostingSpecialistAgent",
    "NeuralSpecialistAgent",
    "ModelSpecialistRegistry",
]
