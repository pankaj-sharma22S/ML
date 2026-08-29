"""Notebook persistence, serialization to standard .ipynb via nbformat, and metadata storage."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from amea.execution.kernel.execution_request import CellType, NotebookCell
from amea.execution.kernel.execution_result import CellExecutionResult, CellOutputType


class NotebookManager:
    """Serializes, loads, and manages .ipynb notebook files and execution metadata."""

    @classmethod
    def save_notebook(
        cls,
        notebook_path: Path | str,
        cells: List[NotebookCell],
        results: Optional[Dict[str, CellExecutionResult]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Export cells and outputs into standard .ipynb format."""
        target_path = Path(notebook_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        nb = new_notebook(metadata=metadata or {"language_info": {"name": "python"}})
        results_map = results or {}

        for c in cells:
            if c.cell_type == CellType.MARKDOWN:
                nb_cell = new_markdown_cell(source=c.source)
            else:
                exec_result = results_map.get(c.cell_id)
                exec_count = exec_result.execution_count if exec_result else c.execution_count
                nb_outputs = []

                if exec_result:
                    for out in exec_result.outputs:
                        if out.output_type == CellOutputType.STREAM:
                            nb_outputs.append(new_output("stream", name=out.stream_name or "stdout", text=out.text or ""))
                        elif out.output_type == CellOutputType.ERROR:
                            nb_outputs.append(new_output("error", ename=out.error_name or "Exception", evalue=out.error_value or "", traceback=out.traceback or []))
                        elif out.output_type == CellOutputType.IMAGE and out.image_base64:
                            nb_outputs.append(new_output("display_data", data={"image/png": out.image_base64}))
                        elif out.output_type in (CellOutputType.TEXT, CellOutputType.SCALAR):
                            nb_outputs.append(new_output("execute_result", data={"text/plain": out.text or str(out.scalar_value)}, execution_count=exec_count))
                        elif out.output_type == CellOutputType.DATAFRAME and out.text:
                            nb_outputs.append(new_output("execute_result", data={"text/plain": out.text}, execution_count=exec_count))

                nb_cell = new_code_cell(source=c.source, execution_count=exec_count, outputs=nb_outputs)

            nb.cells.append(nb_cell)

        with open(target_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return target_path

    @classmethod
    def load_notebook(cls, notebook_path: Path | str) -> List[NotebookCell]:
        """Load cells from an existing .ipynb file."""
        target_path = Path(notebook_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        cells: List[NotebookCell] = []
        for i, c in enumerate(nb.cells):
            cell_type = CellType.MARKDOWN if c.cell_type == "markdown" else CellType.CODE
            cells.append(NotebookCell(
                cell_id=f"cell_{i + 1}",
                cell_type=cell_type,
                source=c.source,
                execution_count=getattr(c, "execution_count", None),
                metadata=dict(getattr(c, "metadata", {})),
            ))

        return cells
