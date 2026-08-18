"""Unit tests for Task and TaskGraph dependency management."""

import pytest
from amea.core.exceptions import TaskDependencyError
from amea.task.model import Task, TaskStatus, TaskPriority
from amea.task.graph import TaskGraph


def test_task_graph_dependency_resolution():
    graph = TaskGraph()
    t1 = Task(task_id="t1", name="Profile Data", required_capability="DATA_PROFILING")
    t2 = Task(task_id="t2", name="Train Model A", required_capability="BOOSTING", dependencies={"t1"})
    t3 = Task(task_id="t3", name="Train Model B", required_capability="LINEAR_ML", dependencies={"t1"})

    graph.add_task(t1)
    graph.add_task(t2)
    graph.add_task(t3)

    # Initially, only t1 is ready
    ready = graph.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].task_id == "t1"

    # Complete t1
    graph.mark_completed("t1", {"profile": "done"})

    # Now both t2 and t3 should be ready in parallel
    ready_after = graph.get_ready_tasks()
    assert len(ready_after) == 2
    assert {t.task_id for t in ready_after} == {"t2", "t3"}


def test_task_graph_cycle_detection():
    graph = TaskGraph()
    t1 = Task(task_id="t1", name="Task 1", required_capability="C1", dependencies={"t2"})
    t2 = Task(task_id="t2", name="Task 2", required_capability="C2", dependencies={"t1"})

    graph.add_task(t1)
    with pytest.raises(TaskDependencyError):
        graph.add_task(t2)
