"""Capability Registry for dynamic, capability-based routing."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class Capability(str, Enum):
    PROBLEM_UNDERSTANDING = "PROBLEM_UNDERSTANDING"
    DATA_PROFILING = "DATA_PROFILING"
    DATA_QUALITY = "DATA_QUALITY"
    EDA = "EDA"
    LEAKAGE_DETECTION = "LEAKAGE_DETECTION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    ML_STRATEGY = "ML_STRATEGY"
    CLASSICAL_ML = "CLASSICAL_ML"
    BOOSTING = "BOOSTING"
    NEURAL_NETWORKS = "NEURAL_NETWORKS"
    TIME_SERIES = "TIME_SERIES"
    EXPERIMENT_RUNNER = "EXPERIMENT_RUNNER"
    EXPERIMENT_TRACKER = "EXPERIMENT_TRACKER"
    EVALUATION = "EVALUATION"
    JUDGE = "JUDGE"
    IMPROVEMENT_PLANNER = "IMPROVEMENT_PLANNER"
    CODE_GENERATION = "CODE_GENERATION"
    CODE_EXECUTION = "CODE_EXECUTION"
    CODE_REPAIR = "CODE_REPAIR"
    SECURITY_VALIDATION = "SECURITY_VALIDATION"


class CapabilityProvider(BaseModel):
    """Metadata for a component providing specific capabilities."""
    name: str
    capabilities: Set[Capability]
    priority: int = 100  # Lower number = higher priority
    is_available: bool = True
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class CapabilityRegistry:
    """Central registry mapping required capabilities to candidate agents/executors."""

    def __init__(self):
        self._providers: Dict[str, CapabilityProvider] = {}
        self._capability_map: Dict[Capability, List[str]] = {cap: [] for cap in Capability}

    def register(self, provider: CapabilityProvider) -> None:
        """Register a new capability provider."""
        self._providers[provider.name] = provider
        for cap in provider.capabilities:
            if provider.name not in self._capability_map[cap]:
                self._capability_map[cap].append(provider.name)
                # Sort by priority
                self._capability_map[cap].sort(key=lambda name: self._providers[name].priority)

    def get_providers_for_capability(self, capability: Capability) -> List[CapabilityProvider]:
        """Find all registered providers capable of providing the specified capability."""
        provider_names = self._capability_map.get(capability, [])
        return [self._providers[name] for name in provider_names if self._providers[name].is_available]

    def get_best_provider(self, capability: Capability) -> Optional[CapabilityProvider]:
        """Get the highest priority available provider for a capability."""
        providers = self.get_providers_for_capability(capability)
        return providers[0] if providers else None

    def list_all_capabilities(self) -> List[Capability]:
        """List all capabilities that currently have at least one available provider."""
        return [cap for cap, names in self._capability_map.items() if any(self._providers[n].is_available for n in names)]

    def set_availability(self, provider_name: str, is_available: bool) -> None:
        """Dynamically enable or disable a provider (e.g. if GPU unavailable or package missing)."""
        if provider_name in self._providers:
            self._providers[provider_name].is_available = is_available
