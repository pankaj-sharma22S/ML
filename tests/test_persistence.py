"""Unit tests for StateCheckpointer and crash recovery."""

from pathlib import Path
import pytest
from amea.core.state import GlobalState, LifecyclePhase, MLTaskSpecification, TaskType
from amea.persistence.checkpointer import StateCheckpointer


def test_checkpoint_save_and_load(tmp_path):
    checkpointer = StateCheckpointer(persistence_dir=tmp_path)
    state = GlobalState(
        project_id="test_proj",
        task_id="task_123",
        user_request="Train model",
        current_phase=LifecyclePhase.INSPECT,
        task_spec=MLTaskSpecification(task_type=TaskType.BINARY_CLASSIFICATION, target_column="y"),
    )

    meta = checkpointer.save_checkpoint(state, checkpoint_id="cp_test_01")
    assert meta.checkpoint_id == "cp_test_01"
    assert meta.phase == "INSPECT"

    # Reload checkpoint
    loaded = checkpointer.load_checkpoint("cp_test_01")
    assert loaded.project_id == "test_proj"
    assert loaded.current_phase == LifecyclePhase.INSPECT
    assert loaded.task_spec.target_column == "y"


def test_checkpoint_list_and_latest(tmp_path):
    checkpointer = StateCheckpointer(persistence_dir=tmp_path)
    state1 = GlobalState(project_id="p", task_id="t", current_phase=LifecyclePhase.UNDERSTAND)
    state2 = GlobalState(project_id="p", task_id="t", current_phase=LifecyclePhase.PLAN)

    checkpointer.save_checkpoint(state1, checkpoint_id="cp_1")
    checkpointer.save_checkpoint(state2, checkpoint_id="cp_2")

    checkpoints = checkpointer.list_checkpoints()
    assert len(checkpoints) == 2

    latest = checkpointer.load_latest_checkpoint()
    assert latest is not None
    assert latest.current_phase == LifecyclePhase.PLAN
