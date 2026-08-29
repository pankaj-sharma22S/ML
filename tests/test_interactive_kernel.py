"""Comprehensive tests for Interactive ML Execution Kernel, Output Parsing, Security, and Notebook Persistence."""

from pathlib import Path
import pytest

from amea.execution.failure_analyzer import FailureCategory
from amea.execution.kernel.ai_cell_assistant import AICellAssistant
from amea.execution.kernel.execution_request import (
    BatchExecuteRequest,
    CellType,
    ExecuteCellRequest,
    NotebookCell,
)
from amea.execution.kernel.execution_result import CellOutputType
from amea.execution.kernel.graph_kernel_executor import GraphKernelExecutor
from amea.execution.kernel.kernel_executor import KernelExecutor
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.kernel.kernel_session import KernelStatus
from amea.execution.kernel.notebook_manager import NotebookManager
from amea.execution.router import (
    AIGenerateCellRequest,
    AIInterpretRequest,
    CreateSessionRequest,
    kernel_router,
)
from amea.task.graph import TaskGraph
from amea.task.model import Task, TaskPriority


@pytest.fixture(scope="module")
def shared_kernel_manager(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("kernel_sandboxes")
    from amea.execution.kernel.kernel_config import KernelConfig
    cfg = KernelConfig(base_workspace_dir=tmp_dir)
    km = KernelManager(config=cfg)
    yield km
    # Cleanup all sessions after module tests
    for s in km.list_sessions():
        km.shutdown(s.session_id)


def test_kernel_lifecycle(shared_kernel_manager):
    session = shared_kernel_manager.create_session(project_id="test_proj_lifecycle")
    assert session.session_id is not None
    assert session.status == KernelStatus.IDLE
    assert shared_kernel_manager.is_alive(session.session_id) is True

    # Test restart
    restarted = shared_kernel_manager.restart(session.session_id)
    assert restarted is True
    assert shared_kernel_manager.is_alive(session.session_id) is True

    # Test shutdown
    shutdown_ok = shared_kernel_manager.shutdown(session.session_id)
    assert shutdown_ok is True
    assert shared_kernel_manager.is_alive(session.session_id) is False


def test_variable_state_persistence_between_cells(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    session = shared_kernel_manager.create_session(project_id="test_proj_vars")

    # Cell 1: Define variables
    req1 = ExecuteCellRequest(
        session_id=session.session_id,
        cell_id="cell_1",
        code="x = 10\ny = 25",
    )
    res1 = executor.execute_cell(req1)
    assert res1.is_success is True

    # Cell 2: Use variables defined in Cell 1
    req2 = ExecuteCellRequest(
        session_id=session.session_id,
        cell_id="cell_2",
        code="x + y",
    )
    res2 = executor.execute_cell(req2)
    assert res2.is_success is True
    scalar_out = next((o for o in res2.outputs if o.output_type == CellOutputType.SCALAR), None)
    assert scalar_out is not None
    assert scalar_out.scalar_value == 35


def test_dataframe_output_table_preview(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    session = shared_kernel_manager.create_session(project_id="test_proj_df")

    code = """import pandas as pd
df = pd.DataFrame({'age': [21, 25, 30], 'salary': [35000, 52000, 68000]})
df
"""
    req = ExecuteCellRequest(
        session_id=session.session_id,
        cell_id="cell_df",
        code=code,
    )
    res = executor.execute_cell(req)
    assert res.is_success is True
    df_out = next((o for o in res.outputs if o.output_type == CellOutputType.DATAFRAME or o.dataframe is not None), None)
    assert df_out is not None
    assert df_out.dataframe is not None
    assert "salary" in df_out.dataframe.columns
    assert len(df_out.dataframe.data) == 3


def test_matplotlib_plot_output(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    session = shared_kernel_manager.create_session(project_id="test_proj_plot")

    code = """import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
"""
    req = ExecuteCellRequest(
        session_id=session.session_id,
        cell_id="cell_plot",
        code=code,
    )
    res = executor.execute_cell(req)
    assert res.is_success is True
    img_out = next((o for o in res.outputs if o.output_type == CellOutputType.IMAGE), None)
    assert img_out is not None
    assert img_out.image_base64 is not None or img_out.image_artifact_path is not None


def test_security_blocks_malicious_code_in_interactive_cell(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    session = shared_kernel_manager.create_session(project_id="test_proj_sec")

    bad_codes = [
        "import subprocess\nsubprocess.run(['dir'])",
        "eval('2 + 2')",
        "import socket\ns = socket.socket()",
        "import os\nos.system('whoami')",
    ]

    for code in bad_codes:
        req = ExecuteCellRequest(
            session_id=session.session_id,
            cell_id="cell_bad",
            code=code,
        )
        res = executor.execute_cell(req)
        assert res.is_success is False
        assert res.status == "SECURITY_BLOCKED"
        assert res.failure_diagnosis is not None
        assert res.failure_diagnosis.category == FailureCategory.SECURITY_VIOLATION


def test_exception_handling_and_failure_analyzer(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    session = shared_kernel_manager.create_session(project_id="test_proj_err")

    req = ExecuteCellRequest(
        session_id=session.session_id,
        cell_id="cell_err",
        code="1 / 0",
    )
    res = executor.execute_cell(req)
    assert res.is_success is False
    assert res.status == "ERROR"
    assert any("ZeroDivisionError" in str(o.error_name) for o in res.outputs)


def test_notebook_save_and_load_ipynb(tmp_path):
    cells = [
        NotebookCell(cell_id="c1", cell_type=CellType.MARKDOWN, source="# Title"),
        NotebookCell(cell_id="c2", cell_type=CellType.CODE, source="print('hello')", execution_count=1),
    ]

    nb_file = tmp_path / "test_notebook.ipynb"
    saved = NotebookManager.save_notebook(notebook_path=nb_file, cells=cells)
    assert saved.exists()

    loaded = NotebookManager.load_notebook(nb_file)
    assert len(loaded) == 2
    assert loaded[0].cell_type == CellType.MARKDOWN
    assert loaded[0].source == "# Title"
    assert loaded[1].source == "print('hello')"


def test_ai_cell_assistant_generation_and_interpretation():
    # 1. AI Cell Generation
    suggestion = AICellAssistant.generate_cell("Check missing values in the dataset")
    assert suggestion.is_safe is True
    assert "isnull().sum()" in suggestion.code
    assert len(suggestion.security_violations) == 0

    # 2. AI Result Interpretation
    from amea.execution.kernel.execution_result import CellExecutionResult, CellOutput
    dummy_res = CellExecutionResult(
        session_id="dummy",
        cell_id="c1",
        status="SUCCESS",
        outputs=[CellOutput(output_type=CellOutputType.STREAM, text="salary: 124 missing")],
        is_success=True,
    )
    interpretation = AICellAssistant.interpret_result(dummy_res)
    assert "completed successfully" in interpretation.summary


def test_graph_kernel_executor(shared_kernel_manager):
    executor = KernelExecutor(shared_kernel_manager)
    graph_exec = GraphKernelExecutor(executor)
    session = shared_kernel_manager.create_session(project_id="test_proj_graph")

    graph = TaskGraph()
    t1 = Task(task_id="t1", name="Data Load", required_capability="data_loading", priority=TaskPriority.HIGH)
    t2 = Task(task_id="t2", name="Feature Prep", required_capability="feature_engineering", dependencies={"t1"}, priority=TaskPriority.NORMAL)
    graph.add_task(t1)
    graph.add_task(t2)

    code_map = {
        "t1": "a = 100",
        "t2": "b = a + 50",
    }

    outcomes = graph_exec.execute_graph_flow(
        session_id=session.session_id,
        graph=graph,
        node_code_map=code_map,
    )

    assert len(outcomes) == 2
    assert outcomes[0].status == "COMPLETED"
    assert outcomes[1].status == "COMPLETED"
    assert graph.is_complete() is True


def test_router_endpoints():
    # Test session creation via router
    sess = kernel_router.create_session(CreateSessionRequest(project_id="router_proj"))
    assert sess.session_id is not None

    # Test AI cell generation via router
    sugg = kernel_router.generate_ai_cell(AIGenerateCellRequest(prompt="Show summary statistics"))
    assert "describe" in sugg.code
    assert sugg.is_safe is True
