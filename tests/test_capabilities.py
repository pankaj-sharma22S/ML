"""Unit tests for CapabilityRegistry and capability-based selection."""

import pytest
from amea.core.capabilities import Capability, CapabilityProvider, CapabilityRegistry


def test_capability_registration_and_lookup():
    registry = CapabilityRegistry()
    p1 = CapabilityProvider(name="LightGBMSpecialist", capabilities={Capability.BOOSTING}, priority=20)
    p2 = CapabilityProvider(name="XGBoostSpecialist", capabilities={Capability.BOOSTING}, priority=10)

    registry.register(p1)
    registry.register(p2)

    # p2 has priority 10 vs p1 priority 20, so p2 should be chosen first
    best = registry.get_best_provider(Capability.BOOSTING)
    assert best is not None
    assert best.name == "XGBoostSpecialist"


def test_dynamic_capability_availability():
    registry = CapabilityRegistry()
    p1 = CapabilityProvider(name="GPUNeuralAgent", capabilities={Capability.NEURAL_NETWORKS}, priority=10)
    registry.register(p1)

    assert registry.get_best_provider(Capability.NEURAL_NETWORKS) is not None

    # Disable when GPU unavailable
    registry.set_availability("GPUNeuralAgent", False)
    assert registry.get_best_provider(Capability.NEURAL_NETWORKS) is None
