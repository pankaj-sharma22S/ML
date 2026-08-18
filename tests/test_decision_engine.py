"""Unit tests for DecisionEngine, gap classification, and rationale."""

import pytest
from amea.core.capabilities import Capability, CapabilityProvider, CapabilityRegistry
from amea.core.state import GlobalState, LifecyclePhase, GapSeverity
from amea.orchestrator.decision_engine import DecisionEngine


def test_decision_engine_gap_detection():
    registry = CapabilityRegistry()
    engine = DecisionEngine(capability_registry=registry)

    state = GlobalState(
        project_id="p",
        task_id="t",
        user_request="Build regression model",
        current_phase=LifecyclePhase.UNDERSTAND,
    )

    decision = engine.evaluate(state)
    assert decision.phase == LifecyclePhase.UNDERSTAND
    assert "ProblemUnderstandingAgent" in [g.affected_components[0] for g in decision.identified_gaps if g.affected_components]
    assert any(g.severity == GapSeverity.CRITICAL for g in decision.identified_gaps)


def test_decision_engine_provider_selection():
    registry = CapabilityRegistry()
    registry.register(CapabilityProvider(name="ProblemUnderstandingAgent", capabilities={Capability.PROBLEM_UNDERSTANDING}, priority=10))
    engine = DecisionEngine(capability_registry=registry)

    state = GlobalState(
        project_id="p",
        task_id="t",
        user_request="Build regression model",
        current_phase=LifecyclePhase.UNDERSTAND,
    )

    decision = engine.evaluate(state)
    assert "ProblemUnderstandingAgent" in decision.selected_agents
    assert decision.confidence_score == 1.0
