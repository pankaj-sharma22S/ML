"""Security-gated interactive Kernel Executor executing code through AMEA security policies."""

import time
from typing import List, Optional

from amea.execution.failure_analyzer import ExecutionFailureAnalyzer, FailureCategory, FailureDiagnosis
from amea.execution.kernel.execution_request import (
    BatchExecuteRequest,
    CellType,
    ExecuteCellRequest,
)
from amea.execution.kernel.execution_result import (
    CellExecutionResult,
    CellOutput,
    CellOutputType,
)
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.security import AstSecurityValidator, DependencySecurityValidator


class KernelExecutor:
    """Coordinates cell execution, AST security validation, resource monitoring, and output diagnostics."""

    def __init__(self, kernel_manager: Optional[KernelManager] = None):
        self.kernel_manager = kernel_manager or KernelManager()
        self.ast_validator = AstSecurityValidator()
        self.dep_validator = DependencySecurityValidator()
        self.failure_analyzer = ExecutionFailureAnalyzer()

    def execute_cell(self, request: ExecuteCellRequest) -> CellExecutionResult:
        """Execute a single notebook cell through full security gate and kernel session."""
        start_time = time.time()

        # 1. AST Security Inspection
        violations = self.ast_validator.validate_code_safety(request.code)
        if violations:
            err_msg = f"Security violation: {'; '.join(violations)}"
            diag = FailureDiagnosis(
                category=FailureCategory.SECURITY_VIOLATION,
                root_cause=err_msg,
                recovery_hint="Modify cell code to remove forbidden modules/calls (e.g. subprocess, os.system, socket, eval).",
                is_retryable=False,
            )
            return CellExecutionResult(
                session_id=request.session_id,
                cell_id=request.cell_id,
                status="SECURITY_BLOCKED",
                execution_count=request.execution_count,
                outputs=[
                    CellOutput(
                        output_type=CellOutputType.ERROR,
                        error_name="SecurityViolationError",
                        error_value=err_msg,
                        text=err_msg,
                    )
                ],
                duration_ms=round((time.time() - start_time) * 1000, 2),
                failure_diagnosis=diag,
                is_success=False,
            )

        # 2. Execute in Python Kernel
        try:
            outputs, exec_count = self.kernel_manager.execute(
                session_id=request.session_id,
                code=request.code,
                timeout_seconds=request.timeout_seconds,
            )
        except TimeoutError as e:
            diag = FailureDiagnosis(
                category=FailureCategory.TIMEOUT,
                root_cause=str(e),
                recovery_hint="Optimize code operations, downsample DataFrame, or increase cell timeout limit.",
                is_retryable=True,
            )
            return CellExecutionResult(
                session_id=request.session_id,
                cell_id=request.cell_id,
                status="TIMEOUT",
                execution_count=request.execution_count,
                outputs=[
                    CellOutput(
                        output_type=CellOutputType.ERROR,
                        error_name="TimeoutError",
                        error_value=str(e),
                        text=str(e),
                    )
                ],
                duration_ms=round((time.time() - start_time) * 1000, 2),
                failure_diagnosis=diag,
                is_success=False,
            )
        except Exception as e:
            err_str = str(e)
            diag = FailureDiagnosis(
                category=FailureCategory.RUNTIME_EXCEPTION,
                root_cause=err_str,
                recovery_hint="Inspect traceback and verify variable existence in kernel session.",
                is_retryable=True,
            )
            return CellExecutionResult(
                session_id=request.session_id,
                cell_id=request.cell_id,
                status="ERROR",
                execution_count=request.execution_count,
                outputs=[
                    CellOutput(
                        output_type=CellOutputType.ERROR,
                        error_name=type(e).__name__,
                        error_value=err_str,
                        text=err_str,
                    )
                ],
                duration_ms=round((time.time() - start_time) * 1000, 2),
                failure_diagnosis=diag,
                is_success=False,
            )

        # 3. Diagnose errors in outputs if any
        error_output = next((o for o in outputs if o.output_type == CellOutputType.ERROR), None)
        diag = None
        status = "SUCCESS"
        is_success = True

        if error_output:
            status = "ERROR"
            is_success = False
            tb_str = "\n".join(error_output.traceback or [])
            full_err = f"{error_output.error_name}: {error_output.error_value}\n{tb_str}"
            diag = self.failure_analyzer._classify_runtime_error(full_err)

        return CellExecutionResult(
            session_id=request.session_id,
            cell_id=request.cell_id,
            status=status,
            execution_count=exec_count,
            outputs=outputs,
            duration_ms=round((time.time() - start_time) * 1000, 2),
            failure_diagnosis=diag,
            is_success=is_success,
        )

    def execute_batch(self, request: BatchExecuteRequest) -> List[CellExecutionResult]:
        """Execute multiple cells sequentially (Run All / Run From Here)."""
        results: List[CellExecutionResult] = []
        started = (request.start_cell_id is None)

        for cell in request.cells:
            if not started:
                if cell.cell_id == request.start_cell_id:
                    started = True
                else:
                    continue

            if cell.cell_type != CellType.CODE:
                continue

            cell_req = ExecuteCellRequest(
                session_id=request.session_id,
                cell_id=cell.cell_id,
                code=cell.source,
                timeout_seconds=request.timeout_per_cell_seconds,
            )
            res = self.execute_cell(cell_req)
            results.append(res)

            if not res.is_success and request.stop_on_error:
                break

        return results
