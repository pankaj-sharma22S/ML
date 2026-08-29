"""API Router and dispatcher for Interactive Kernel sessions and Notebook execution."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.execution.kernel.ai_cell_assistant import AICellAssistant, AICellSuggestion, AIInterpretation
from amea.execution.kernel.execution_request import BatchExecuteRequest, ExecuteCellRequest, NotebookCell
from amea.execution.kernel.execution_result import CellExecutionResult
from amea.execution.kernel.kernel_executor import KernelExecutor
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.kernel.kernel_session import KernelSession
from amea.execution.kernel.notebook_manager import NotebookManager


class CreateSessionRequest(BaseModel):
    project_id: str = "default_project"
    session_id: Optional[str] = None


class SessionActionRequest(BaseModel):
    session_id: str


class SaveNotebookRequest(BaseModel):
    notebook_path: str
    cells: List[NotebookCell]
    metadata: Optional[Dict[str, Any]] = None


class LoadNotebookRequest(BaseModel):
    notebook_path: str


class AIGenerateCellRequest(BaseModel):
    prompt: str
    active_variables: Optional[List[str]] = None


class AIInterpretRequest(BaseModel):
    result: CellExecutionResult


class InteractiveKernelRouter:
    """REST dispatcher for interactive notebook kernels, execution, and AI assistance."""

    def __init__(self, prefix: str = "/api/kernel"):
        self.prefix = prefix
        self.kernel_manager = KernelManager()
        self.kernel_executor = KernelExecutor(self.kernel_manager)

    def create_session(self, request: CreateSessionRequest) -> KernelSession:
        return self.kernel_manager.create_session(
            project_id=request.project_id,
            session_id=request.session_id,
        )

    def get_session(self, session_id: str) -> Optional[KernelSession]:
        return self.kernel_manager.get_session(session_id)

    def execute_cell(self, request: ExecuteCellRequest) -> CellExecutionResult:
        return self.kernel_executor.execute_cell(request)

    def execute_batch(self, request: BatchExecuteRequest) -> List[CellExecutionResult]:
        return self.kernel_executor.execute_batch(request)

    def interrupt_session(self, request: SessionActionRequest) -> Dict[str, bool]:
        success = self.kernel_manager.interrupt(request.session_id)
        return {"success": success}

    def restart_session(self, request: SessionActionRequest) -> Dict[str, bool]:
        success = self.kernel_manager.restart(request.session_id)
        return {"success": success}

    def shutdown_session(self, session_id: str) -> Dict[str, bool]:
        success = self.kernel_manager.shutdown(session_id)
        return {"success": success}

    def list_sessions(self, project_id: Optional[str] = None) -> List[KernelSession]:
        return self.kernel_manager.list_sessions(project_id)

    def save_notebook(self, request: SaveNotebookRequest) -> Dict[str, str]:
        path = NotebookManager.save_notebook(
            notebook_path=request.notebook_path,
            cells=request.cells,
            metadata=request.metadata,
        )
        return {"saved_path": str(path)}

    def load_notebook(self, request: LoadNotebookRequest) -> List[NotebookCell]:
        return NotebookManager.load_notebook(request.notebook_path)

    def generate_ai_cell(self, request: AIGenerateCellRequest) -> AICellSuggestion:
        return AICellAssistant.generate_cell(
            user_prompt=request.prompt,
            active_variables=request.active_variables,
        )

    def interpret_result(self, request: AIInterpretRequest) -> AIInterpretation:
        return AICellAssistant.interpret_result(request.result)


# Global singleton instance for interactive kernel router
kernel_router = InteractiveKernelRouter()
