"""Failure-oriented tests for invalid transitions, missing data, timeouts, and budget limits."""

from pathlib import Path
import pytest
from amea.core.config import ComputeBudget, ProjectConfig
from amea.core.events import EventBus
from amea.core.state import GlobalState, LifecyclePhase, MLTaskSpecification, TaskType
from amea.execution.subprocess_executor import SubprocessExecutor
from amea.orchestrator.decision_engine import DecisionEngine
from amea.orchestrator.nodes import OrchestratorNodes
from amea.orchestrator.workflow import OrchestratorWorkflow
from amea.persistence.checkpointer import StateCheckpointer
from amea.core.capabilities import CapabilityRegistry


def test_missing_dataset_termination(tmp_path):
    nodes = OrchestratorNodes()
    state = GlobalState(
        project_id="fail_test",
        task_id="t1",
        user_request="Train model",
        current_phase=LifecyclePhase.INSPECT,
        dataset_metadata={"dataset_path": str(tmp_path / "non_existent_file.csv")},
    )

    result_state = nodes.inspect_node(state)
    assert result_state.current_phase == LifecyclePhase.TERMINATED
    assert result_state.is_terminal
    assert "does not exist" in (result_state.termination_reason or "")


def test_missing_target_termination():
    nodes = OrchestratorNodes()
    state = GlobalState(
        project_id="fail_test",
        task_id="t1",
        current_phase=LifecyclePhase.VALIDATE,
        task_spec=MLTaskSpecification(task_type=TaskType.BINARY_CLASSIFICATION, target_column="non_existent_col"),
        # data profile without that target column
        data_profile=None,
    )

    result_state = nodes.validate_node(state)
    assert result_state.current_phase == LifecyclePhase.TERMINATED
    assert result_state.is_terminal


def test_execution_timeout_handling(tmp_path):
    executor = SubprocessExecutor(sandbox_root=tmp_path)
    # Script that sleeps longer than timeout
    script = """
import time
time.sleep(10)
"""
    res = executor.execute_script(run_id="timeout_test", script_content=script, timeout_seconds=1)
    assert not res.is_success
    assert res.error_type == "TimeoutError"
    assert "TIMEOUT" in res.stderr
