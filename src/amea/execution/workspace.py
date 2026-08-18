"""Isolated workspace management for individual task/experiment runs."""

import os
import shutil
from pathlib import Path
from typing import Optional


class IsolatedWorkspace:
    """Creates and manages an ephemeral, isolated filesystem directory for a worker run."""

    def __init__(self, base_dir: Path, run_id: str):
        self.run_id = run_id
        self.workspace_path = (base_dir / f"run_{run_id}").resolve()

    def create(self) -> Path:
        """Create the isolated workspace directory."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        return self.workspace_path

    def write_file(self, relative_path: str, content: str) -> Path:
        """Safely write a file inside the isolated workspace."""
        target = (self.workspace_path / relative_path).resolve()
        # Security check: Ensure file does not escape workspace
        if not str(target).startswith(str(self.workspace_path)):
            raise ValueError(f"Path traversal blocked: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_file(self, relative_path: str) -> Optional[str]:
        """Safely read a file from the isolated workspace."""
        target = (self.workspace_path / relative_path).resolve()
        if not str(target).startswith(str(self.workspace_path)):
            raise ValueError(f"Path traversal blocked: {relative_path}")
        if target.exists():
            return target.read_text(encoding="utf-8")
        return None

    def cleanup(self, preserve: bool = False) -> None:
        """Remove the workspace unless preserve flag is set."""
        if not preserve and self.workspace_path.exists():
            shutil.rmtree(self.workspace_path, ignore_errors=True)
