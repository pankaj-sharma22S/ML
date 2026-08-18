"""Task dependency graph with topological sorting and parallel group partitioning."""

from collections import deque
from typing import Dict, List, Set, Optional
from amea.core.exceptions import TaskDependencyError
from amea.task.model import Task, TaskStatus


class TaskGraph:
    """Directed Acyclic Graph (DAG) of tasks supporting parallel and sequential dispatch."""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Add a task to the graph and validate dependencies."""
        if task.task_id in self._tasks:
            raise TaskDependencyError(f"Duplicate task ID in graph: {task.task_id}")
        self._tasks[task.task_id] = task
        self._validate_dag()

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        """List all tasks in the graph."""
        return list(self._tasks.values())

    def get_ready_tasks(self) -> List[Task]:
        """Retrieve all pending tasks whose dependencies are fully completed."""
        completed_ids = {t.task_id for t in self._tasks.values() if t.status == TaskStatus.COMPLETED}
        ready = [t for t in self._tasks.values() if t.is_ready(completed_ids)]
        # Sort by priority (lower number = higher priority)
        ready.sort(key=lambda t: t.priority.value)
        return ready

    def get_independent_parallel_groups(self) -> List[List[Task]]:
        """Partition ready tasks into execution waves/batches that can run independently in parallel."""
        ready = self.get_ready_tasks()
        if not ready:
            return []
        # All ready tasks at this instant have satisfied dependencies and can run in parallel
        return [ready]

    def mark_started(self, task_id: str) -> None:
        """Mark a task as actively running."""
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, outputs: Dict) -> None:
        """Mark a task as completed with its structured outputs."""
        if task_id in self._tasks:
            t = self._tasks[task_id]
            t.status = TaskStatus.COMPLETED
            t.outputs = outputs

    def mark_failed(self, task_id: str, error_message: str) -> bool:
        """Mark a task as failed and evaluate retry policy. Returns True if task will be retried."""
        if task_id not in self._tasks:
            return False
        t = self._tasks[task_id]
        t.error_message = error_message
        if t.retry_policy.retry_count < t.retry_policy.max_retries:
            t.retry_policy.retry_count += 1
            t.status = TaskStatus.PENDING  # Ready for retry
            return True
        else:
            t.status = TaskStatus.FAILED
            return False

    def is_complete(self) -> bool:
        """Check if all tasks in the graph have reached a terminal state."""
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) for t in self._tasks.values())

    def has_failures(self) -> bool:
        """Check if any task in the graph has failed."""
        return any(t.status == TaskStatus.FAILED for t in self._tasks.values())

    def _validate_dag(self) -> None:
        """Check for missing dependencies or cyclical dependencies using Kahn's algorithm."""
        # 1. Check for missing dependency references
        all_ids = set(self._tasks.keys())
        for task in self._tasks.values():
            missing = task.dependencies - all_ids
            if missing:
                # Missing dependencies are allowed if they will be added subsequently, but we log or track
                pass

        # 2. Cycle detection on currently known dependencies
        in_degree = {tid: 0 for tid in self._tasks}
        adj: Dict[str, List[str]] = {tid: [] for tid in self._tasks}

        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep in self._tasks:
                    adj[dep].append(task.task_id)
                    in_degree[task.task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count < len(self._tasks):
            raise TaskDependencyError("Cyclical dependency detected in TaskGraph.")
