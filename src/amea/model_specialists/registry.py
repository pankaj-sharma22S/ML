"""Capability-based Registry for Model Specialist Agents."""

from typing import Dict, List, Optional
from amea.ml_strategy.models import ModelFamily
from amea.model_specialists.base import ModelCapability, ModelSpecialistBase
from amea.model_specialists.boosting_specialist import BoostingSpecialistAgent
from amea.model_specialists.linear_specialist import LinearSpecialistAgent
from amea.model_specialists.neural_specialist import NeuralSpecialistAgent
from amea.model_specialists.tree_specialist import TreeModelSpecialistAgent


class ModelSpecialistRegistry:
    """Registry maintaining capability-isolated Model Specialist Agents."""

    def __init__(self):
        self._specialists: Dict[str, ModelSpecialistBase] = {
            ModelFamily.LINEAR_MODEL.value: LinearSpecialistAgent(),
            ModelFamily.RANDOM_FOREST.value: TreeModelSpecialistAgent(),
            ModelFamily.GRADIENT_BOOSTING.value: BoostingSpecialistAgent(),
            ModelFamily.TABULAR_NEURAL_NET.value: NeuralSpecialistAgent(),
            "LinearModel": LinearSpecialistAgent(),
            "RandomForest": TreeModelSpecialistAgent(),
            "GradientBoosting": BoostingSpecialistAgent(),
            "TabularNeuralNet": NeuralSpecialistAgent(),
        }

    def get_specialist(self, model_family: str | ModelFamily) -> Optional[ModelSpecialistBase]:
        key = model_family.value if isinstance(model_family, ModelFamily) else str(model_family)
        return self._specialists.get(key)

    def list_capabilities(self) -> List[ModelCapability]:
        unique_specialists = {id(s): s for s in self._specialists.values()}.values()
        return [s.capabilities() for s in unique_specialists]
