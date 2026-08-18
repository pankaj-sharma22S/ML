"""Unit tests for GlobalState, schemas, ownership, and validated reducers."""

import pytest
from pydantic import ValidationError

from amea.core.exceptions import InvalidTransitionError, StateOwnershipError
from amea.core.reducers import StatePatch, apply_state_patch
from amea.core.state import GlobalState, LifecyclePhase, MLTaskSpecification, TaskType


def test_global_state_initialization():
    """Verify GlobalState initializes with correct defaults."""
    state = GlobalState(project_id="p1", task_id="t1", user_request="Train classifier")
    assert state.current_phase == LifecyclePhase.UNDERSTAND
    assert state.schema_version == "1.0.0"
    assert len(state.experiment_ledger) == 0


def test_valid_state_patch_application():
    """Verify valid patch updates state correctly."""
    state = GlobalState(project_id="p1", task_id="t1")
    spec = MLTaskSpecification(task_type=TaskType.BINARY_CLASSIFICATION, target_column="target")
    patch = StatePatch(
        author_component="ProblemUnderstandingAgent",
        task_spec=spec,
        target_phase=LifecyclePhase.INSPECT,
    )
    new_state = apply_state_patch(state, patch)
    assert new_state.current_phase == LifecyclePhase.INSPECT
    assert new_state.task_spec is not None
    assert new_state.task_spec.target_column == "target"


def test_invalid_phase_transition_rejected():
    """Verify illegal lifecycle transition raises InvalidTransitionError."""
    state = GlobalState(project_id="p1", task_id="t1", current_phase=LifecyclePhase.UNDERSTAND)
    # Trying to jump directly from UNDERSTAND to BUILD
    patch = StatePatch(
        author_component="Orchestrator",
        target_phase=LifecyclePhase.BUILD,
    )
    with pytest.raises(InvalidTransitionError):
        apply_state_patch(state, patch)


def test_unauthorized_state_ownership_rejected():
    """Verify component cannot modify state fields it does not own."""
    state = GlobalState(project_id="p1", task_id="t1", current_phase=LifecyclePhase.UNDERSTAND)
    spec = MLTaskSpecification(task_type=TaskType.BINARY_CLASSIFICATION, target_column="target")
    # DataProfiler is not permitted to write task_spec
    patch = StatePatch(
        author_component="DataProfiler",
        task_spec=spec,
    )
    with pytest.raises(StateOwnershipError):
        apply_state_patch(state, patch)
