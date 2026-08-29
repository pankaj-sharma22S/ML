"""Executes visual TaskGraph DAG nodes inside the interactive kernel session."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from amea.execution.kernel.execution_request import ExecuteCellRequest
from amea.execution.kernel.execution_result import CellExecutionResult
from amea.execution.kernel.kernel_executor import KernelExecutor
from amea.task.graph import TaskGraph
from amea.task.model import Task, TaskStatus


class NodeExecutionOutcome(BaseModel):
    """Result of running a visual workflow node in the interactive kernel."""
    task_id: str
    status: TaskStatus
    cell_result: CellExecutionResult
    duration_ms: float = 0.0


class GraphKernelExecutor:
    """Bridges visual TaskGraph nodes to the interactive Python kernel session."""

    def __init__(self, kernel_executor: Optional[KernelExecutor] = None):
        self.kernel_executor = kernel_executor or KernelExecutor()

    def execute_node(
        self,
        session_id: str,
        task: Task,
        code_to_execute: str,
    ) -> NodeExecutionOutcome:
        """Execute a single workflow graph node code in the shared interactive kernel."""
        req = ExecuteCellRequest(
            session_id=session_id,
            cell_id=task.task_id,
            code=code_to_execute,
            timeout_seconds=task.resources.timeout_seconds if task.resources else None,
        )

        res = self.kernel_executor.execute_cell(req)

        if res.is_success:
            task.status = TaskStatus.COMPLETED
            task.outputs = {"metrics": res.outputs, "execution_count": res.execution_count}
            status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
            task.error_message = res.failure_diagnosis.root_cause if res.failure_diagnosis else "Node execution error"
            status = TaskStatus.FAILED

        return NodeExecutionOutcome(
            task_id=task.task_id,
            status=status,
            cell_result=res,
            duration_ms=res.duration_ms,
        )

    def execute_graph_flow(
        self,
        session_id: str,
        graph: TaskGraph,
        node_code_map: Dict[str, str],
    ) -> List[NodeExecutionOutcome]:
        """Execute an entire task graph in topological order through the kernel."""
        outcomes: List[NodeExecutionOutcome] = []

        while not graph.is_complete():
            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            for t in ready_tasks:
                code = node_code_map.get(t.task_id, f"# Node {t.task_id}\npass")
                graph.mark_started(t.task_id)
                outcome = self.execute_node(session_id, t, code)
                outcomes.append(outcome)

                if outcome.status == TaskStatus.COMPLETED:
                    graph.mark_completed(t.task_id, t.outputs)
                else:
                    graph.mark_failed(t.task_id, outcome.cell_result.failure_diagnosis.root_cause if outcome.cell_result.failure_diagnosis else "Execution failed")
                    return outcomes

        return outcomes
